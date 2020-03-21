#include "activity_predictor.h"
#include "constants.h"

int PredictActivity(float *output) {
  int predicted = -1;
  for (int i = 0; i < activity_num_classes; i++) {
    if (output[i] > activity_prediction_threshold) {
      predicted = i;
    }
  }

  return predicted;
}
