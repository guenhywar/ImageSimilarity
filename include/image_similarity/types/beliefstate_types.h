/*
 * This file was generated by generate_typesystem.py.
 * Filename:    beliefstate_types
 * Name:        beliefstate
 * Description: No description given
 * Version:     1.0
 * Vendor:      None
 */
#ifndef __BELIEFSTATE_TYPES_H__
#define __BELIEFSTATE_TYPES_H__

#include <robosherlock/feature_structure_proxy.h>
#include <image_similarity/types/type_definitions.h>
#include <robosherlock/types/scene_types.h>

namespace image_similarity
{

/*
 * Data from the internal belief state that corresponds to an object hypothesis
 */
class BeliefStateObject : public rs::Object
{
private:
  void initFields()
  {
    simulationActorId.init(this, "simulationActorId");
    rsObjectId.init(this, "rsObjectId");
    mask_color_r.init(this, "mask_color_r");
    mask_color_g.init(this, "mask_color_g");
    mask_color_b.init(this, "mask_color_b");
  }
public:
  // This is the ID that the simulation engine uses internally to represent the object.
  rs::FeatureStructureEntry<std::string> simulationActorId;
  // This is the ID we assigned to the object in the simulator. rsObjectId is usually not equal to simulationActorId
  rs::FeatureStructureEntry<std::string> rsObjectId;
  // Red Color of this object in the object mask
  rs::FeatureStructureEntry<int> mask_color_r;
  // Green Color of this object in the object mask
  rs::FeatureStructureEntry<int> mask_color_g;
  // Blue Color of this object in the object mask
  rs::FeatureStructureEntry<int> mask_color_b;

  BeliefStateObject(const BeliefStateObject &other) :
      rs::Object(other)
  {
    initFields();
  }

  BeliefStateObject(uima::FeatureStructure fs) :
      rs::Object(fs)
  {
    initFields();
  }
};

}

TYPE_TRAIT(image_similarity::BeliefStateObject, IMAGE_SIMILARITY_BELIEFSTATE_BELIEFSTATEOBJECT)

#endif /* __BELIEFSTATE_TYPES_H__ */