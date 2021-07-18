#include <uima/api.hpp>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>
#include <random>

using namespace uima;

//classifier to get a random classification result withouth real knowlede. its just random as baseline
class baselineClassifier : public DrawingAnnotator {

public:
    baselineClassifier() : DrawingAnnotator(__func__) {

    }

    TyErrorId initialize(AnnotatorContext &ctx) {
        outInfo("initialize");
        return UIMA_ERR_NONE;
    }

    TyErrorId destroy() {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec) {

        outInfo("process start");
        rs::StopWatch clock;
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();

        std::vector<rs::ObjectHypothesis> clusters;
        scene.identifiables.filter(clusters);

        for(size_t i = 0; i < clusters.size(); ++i) {
            rs::ObjectHypothesis &cluster = clusters[i];

            std::vector<std::string> possibleClassifications;
            possibleClassifications.push_back("AlbiHimbeerJuice");
            possibleClassifications.push_back("KoellnMuesliKnusperHonigNuss");
            possibleClassifications.push_back("SojaMilch");
            possibleClassifications.push_back("PfannerGruneIcetea");
            possibleClassifications.push_back("PfannerPfirsichIcetea");

            //from https://stackoverflow.com/questions/6926433/how-to-shuffle-a-stdvector
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);
            std::shuffle(possibleClassifications.begin(), possibleClassifications.end(),e);

            float fakeConfidence = 1.0;

            //fill the cas with faked object classification results in random order
            for (std::string classification : possibleClassifications) {
                rs::Classification classResult = rs::create<rs::Classification>(tcas);
                classResult.classname.set(classification);
                classResult.classifier("baselineClassifier");
                classResult.source.set("baseline");
                rs::ClassConfidence confidence = rs::create<rs::ClassConfidence>(tcas);
                confidence.score.set(fakeConfidence);
                confidence.name.set(classification);
                classResult.confidences.set({confidence});
                cluster.annotations.append(classResult);
                fakeConfidence = fakeConfidence - 0.01;
            }
        }
        outInfo("took: " << clock.getTime() << " ms.");
        return UIMA_ERR_NONE;
    }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(baselineClassifier)
