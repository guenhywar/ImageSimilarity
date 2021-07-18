#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <opencv2/ml.hpp>

//#include <robosherlock/types/all_types.h>
//RS
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>
#include <robosherlock/CASConsumerContext.h>

//imsim
#include <image_similarity/types/similarity_types.h>

using namespace uima;


class SimilaritySummarizer : public Annotator {
private:
    std::ofstream myFile;
    std::string other_cas_id;
    int rounds = 0;
    int min_samples_split;
    int max_depth;
    int max_features;
    std::string training_data_path;
    std::string classify_data_path;
    std::string ground_truth_path;


public:
    cv::Ptr<cv::ml::RTrees> rndTree;

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        if (ctx.isParameterDefined("otherCASId")) {
            ctx.extractValue("otherCASId", other_cas_id);
            outInfo("Using AAE/CAS identified by '" << other_cas_id << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("min_samples_split")) {
            ctx.extractValue("min_samples_split", min_samples_split);
            outInfo("Using min_samples_split '" << min_samples_split << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("max_depth")) {
            ctx.extractValue("max_depth", max_depth);
            outInfo("Using max_depth '" << max_depth << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("max_features")) {
            ctx.extractValue("max_features", max_features);
            outInfo("Using max_features '" << max_features << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("training_data")) {
            ctx.extractValue("training_data", training_data_path);
            outInfo("Using training_data '" << training_data_path << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("classify_data")) {
            ctx.extractValue("classify_data", classify_data_path);
            outInfo("Using classify_data '" << classify_data_path << "' for imagesimilarity");
        }
        if (ctx.isParameterDefined("ground_truth")) {
            ctx.extractValue("ground_truth", ground_truth_path);
            outInfo("Using ground_truth '" << ground_truth_path << "' for imagesimilarity");
        }


        myFile.open("/home/ros/Desktop/similarity.csv", std::ofstream::out | std::ofstream::app);
        myFile << "Classification" << "," << "Compared to" << "," << "identical" << "," << "SurfMatches" << ","
               << "PCL Matches" << "," << "ICP Similarityscore" << "," << "ColorHistDistance" << "," << "guess" << ","
               << "neighborsWithDist" << std::endl;
        myFile.close();
        loadTrainingDataAndTrain();
        validateTraining();

        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    void loadTrainingDataAndTrain() {

        cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::loadFromCSV(training_data_path,
                                                                          1, 0, 1);

        rndTree = cv::ml::RTrees::create();


        rndTree->setMaxDepth(max_depth);
        rndTree->setMinSampleCount(min_samples_split);
        rndTree->setMaxCategories(max_features);
        rndTree->train(tdata);


    }

    void validateTraining() {

        std::ifstream trainDataStream;

        cv::Mat results;
        cv::Ptr<cv::ml::TrainData> pdata = cv::ml::TrainData::loadFromCSV(classify_data_path, 1, 0, 1);
        cv::Mat y_predict;
        results = rndTree->predict(pdata->getSamples(), y_predict);
        trainDataStream.open(ground_truth_path);


        int true_positive = 0;
        int true_negative = 0;
        int false_positive = 0;
        int false_negative = 0;
        std::string line;
        int val;
        // Keep track of the current column index
        int colIdx = 0;

        while (std::getline(trainDataStream, line)) {
            // Create a stringstream of the current line
            std::stringstream ss(line);

            // Extract each integer
            while (ss >> val) {
                // outInfo("val:" << val);
                //  outInfo("pred: " << std::to_string(y_predict.at<float>(colIdx, 0)));
                int pred = y_predict.at<float>(colIdx, 0);
                if (val == 1 and pred == 1) {
                    true_positive++;
                }
                if (val == 0 and pred == 0) {
                    true_negative++;
                }
                if (val == 0 and pred == 1) {
                    false_positive++;
                }
                if (val == 1 and pred == 0) {
                    false_negative++;
                }
            }
            colIdx++;
        }

        outInfo("Confusionmatrix for Random Forest Training in Similarity Summarizer:");
        outInfo("    predneg    predpos     ");
        outInfo("neg    " << true_negative << "         " << false_positive);
        outInfo("pos    " << false_negative << "        " << true_positive);

        float right_ones = true_negative + true_positive;
        float false_ones = false_positive + false_negative;
        float all = right_ones + false_ones;
        float accuracy_tree = right_ones / all;

        outInfo("Accuracy:  " << accuracy_tree);


    }


    TyErrorId process(CAS
                      &tcas,
                      ResultSpecification const &res_spec    ) {

        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();


        uima::CAS *other_cas;
        other_cas = rs::CASConsumerContext::getInstance().getCAS(other_cas_id);
        if (!other_cas) {
            outError("Couldn't fetch CAS identified by '"
                             << other_cas_id
                             << "'. Make sure you have loaded an AAE with that name and "
                             << " that you've set 'otherCASId' in this config");
            return UIMA_ERR_ANNOTATOR_MISSING_INFO;
        }
        rs::SceneCas other_cas_scenecas(*other_cas);
        rs::Scene other_cas_scene = other_cas_scenecas.getScene();

        std::string neighborsAndDistances = "";
        std::vector<std::string> foundClasses;

        //get knnneighbors via othercas
        std::vector<rs::ObjectHypothesis> oh;
        other_cas_scene.identifiables.
                filter(oh);
        if (oh.size() == 0) {
            outInfo("No ObjectHypothesis found.");
        }
        for (auto a : oh) {
            std::vector<rs::Classification> classifiedNeighbors;
            a.annotations.
                    filter(classifiedNeighbors);
            if (classifiedNeighbors.size() == 0) {
                outInfo("No Classification found.");
            }
            for (rs::Classification cn : classifiedNeighbors) {
                for (auto conf  : cn.confidences.get()) {
                    outInfo(conf.name.get());
                    outInfo(conf.score.get());
                    if (std::find(foundClasses.begin(), foundClasses.end(), conf.name.get()) == foundClasses.end()) {
                        foundClasses.push_back(conf.name.get());
                        neighborsAndDistances =
                                neighborsAndDistances + conf.name.get() + " " + std::to_string(conf.score.get()) + "; ";
                    }
                }
            }
        }


        myFile.open("/home/ros/Desktop/similarity.csv", std::ofstream::out | std::ofstream::app);

        std::vector<image_similarity::SimilarityObject> imsimobjects;
        scene.annotations.filter(imsimobjects);
        if (imsimobjects.size() == 0) {
            outInfo("No objects found.");
        }


//atm nur ein objekt drin und für diese pipeline auch das einzig mögliche (sonst würde der quatsch mit den runden nicht gehen
        for (image_similarity::SimilarityObject iso  : imsimobjects) {
            rounds++;
std::string comparedTo = "";
           // std::string comparedTo = "KoellnMuesliKnusperHonigNuss";
            bool imres = false;
 if (rounds <= 10) {
     comparedTo = "KoellnMuesliKnusperHonigNuss";
     //imres = true;
 }
 if (rounds > 10 && rounds <= 20) {
     comparedTo = "AlbiHimbeerJuice";
     //imres = false;
 }
 if (rounds > 20 && rounds <= 30) {
     comparedTo = "KelloggsCornFlakes";
     //imres = false;
 }
 if (rounds > 30 && rounds <= 40) {
     comparedTo = "VollMilch";
     //imres = false;
 }
 if (rounds > 40 && rounds <= 50) {
     comparedTo = "SpitzenReis";
     //imres = false;
 }
 if (rounds > 50 && rounds <= 60) {
     comparedTo = "PfannerGruneIcetea";
     //imres = false;
 }
 if (rounds > 60 && rounds <= 70) {
     comparedTo = "SojaMilch";
     //imres = false;
 }

            if (comparedTo == iso.classification.get()) {
                imres = true;
            }

            int resultCounter = 0;
            bool result = false;
            outInfo(iso.matches_surf.get() << " " << iso.matches_pcl.get() << " " << iso.similarityscore_icp.get() << " " <<  iso.colorHist_distance.get() );
           // cv::Mat dataToPredictOn;

            float data[4] = { float(iso.matches_surf.get()), float(iso.matches_pcl.get()), iso.similarityscore_icp.get(), iso.colorHist_distance.get()};
            cv::Mat dataToPredictOn = cv::Mat(1, 4, CV_32F, data);

            cv::Mat prediction;
            rndTree->predict(dataToPredictOn, prediction);
            int resultAsInt = prediction.at<float>(0,0);
            if(resultAsInt == 1) {
                result = true;
            } else {
                result= false;
            }
           /* if (iso.matches_surf.get() >= 20) { //40
                resultCounter = resultCounter + 1;
            }
            if (iso.matches_pcl.get() > 5) { // 6
                resultCounter = resultCounter + 1;
            }
            if (iso.similarityscore_icp.get() <= 0.00002) { //0.00001
                resultCounter = resultCounter + 1;
            }

            if (iso.colorHist_distance.get() <= 0.76) { //0.79
                resultCounter = resultCounter + 1;
            }

            if (resultCounter >= 3) {
                result = true;
            }*/


// bool imres = true;
//std::string comparedTo = "KoellnMuesliKnusperHonigNuss";

//iso.classification.get(); //KoellnMuesliKnusperHonigNuss
//"AlbiHimbeerJuice";
//"KelloggsCornFlakes";
//"VollMilch";
//"SpitzenReis";
//"PfannerGruneIcetea";
//"SojaMilch";
            outInfo("classification ");
            outInfo(iso.classification.get());
            outInfo("Compared_to");
            outInfo(comparedTo);
            outInfo("Surfmatches ");
            outInfo(iso.matches_surf.get());
            outInfo("PClmatches ");
            outInfo(iso.matches_pcl.get());
            outInfo("ICP ");
            outInfo(iso.similarityscore_icp.get());
            outInfo("ColorHistDist ");
            outInfo(iso.colorHist_distance.get());
            outInfo("guess ");
            outInfo(result);
            outInfo("neighborsWithDist ");
            outInfo(neighborsAndDistances);
            myFile << iso.classification.get() << "," << comparedTo << "," << imres << "," << iso.matches_surf.get()
                   << "," << iso.matches_pcl.get() << "," << iso.similarityscore_icp.get() << ","
                   << iso.colorHist_distance.get() << "," << result << "," << neighborsAndDistances << std::endl;
        }
        myFile.close();


        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;

    }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(SimilaritySummarizer)
