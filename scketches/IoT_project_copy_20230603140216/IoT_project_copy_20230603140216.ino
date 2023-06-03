/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <Arduino_LSM9DS1.h>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
int offset = 0;

constexpr int kTensorArenaSize = 3000; //2000
uint8_t tensor_arena[kTensorArenaSize];


float mean_[60] = {-0.00260862, 0.00463733, 0.99483616, -0.00328056, 0.00515832, 0.99825479
                , -0.00427283, 0.00368925, 0.98173417, -0.00300994, 0.00294735, 0.97419183
                , -0.00282032, 0.00372054, 0.98384389, -0.00317931, 0.00451583, 0.99311672
                , -0.00368557, 0.00401141, 0.98819219, -0.00383837, 0.00358616, 0.97888071
                , -0.00293078, 0.00348675, 0.98062592, -0.00256996, 0.00403166, 0.98941274
                , -0.00328424, 0.00421024, 0.99001105, -0.00346281, 0.00366716, 0.98226252
                , -0.00327688, 0.00327872, 0.97845361, -0.00293814, 0.00393962, 0.9862813
                , -0.00259941, 0.00455633, 0.99490059, -0.00293078, 0.00405007, 0.98596465
                , -0.00326583, 0.00284978, 0.97321429, -0.00307806, 0.00344072, 0.97881627
                , -0.00314433, 0.00519146, 1.00297128, -0.00290685, 0.0050313, 0.99771907};

float scale_[60] = {0.01059476, 0.01482771, 0.04470953, 0.01067615, 0.01526318, 0.04630503, 
                    0.00972304, 0.01455822, 0.04397931, 0.00964662, 0.01433037, 0.04667646,
                    0.01030279, 0.01451922, 0.04201861, 0.01060915, 0.01499709, 0.04588464,
                    0.01044671, 0.01467801, 0.0433052, 0.01005572, 0.01450929, 0.04486889,
                    0.01023061, 0.01440372, 0.04312591, 0.01052491, 0.01471943, 0.04612652,
                    0.01051692, 0.01475881, 0.04462163, 0.01014044, 0.01459284, 0.04479559,
                    0.01057333, 0.01440081, 0.04450846, 0.01117628, 0.01475931, 0.04760788,
                    0.01085193, 0.01501675, 0.04786253, 0.00960124, 0.01478303, 0.04485524,
                    0.01037287, 0.01426711, 0.04413019, 0.01217459, 0.01468993, 0.05021802,
                    0.01186645, 0.01540327, 0.0510343, 0.01022934, 0.01535655, 0.04845598};



}  // namespace




// The name of this function is important for Arduino compatibility.
void setup() {
    Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
  offset = 0;



  Serial.print(input->dims->size);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float x, y, z;
  
  if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x, y, z);

      //x,y,z läser in korrekt
      float scaled_x = (x- mean_[0 + offset*3]) / scale_[0 + offset*3];
      float scaled_y = (y- mean_[1 + offset*3]) / scale_[1 + offset*3];
      float scaled_z = (z- mean_[2 + offset*3]) / scale_[2 + offset*3];

      int8_t x_quantized = scaled_x / input->params.scale + input->params.zero_point;
      int8_t y_quantized = scaled_y / input->params.scale + input->params.zero_point;
      int8_t z_quantized = scaled_z / input->params.scale + input->params.zero_point;

  
      input->data.int8[0 + offset*3] = x_quantized;
      input->data.int8[1 + offset*3] = y_quantized;
      input->data.int8[2 + offset*3] = z_quantized;

      offset++;

      if (offset >= 20) {
        offset = 0;
      } else {
        return;
      }


      // Run inference, and report any error
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                            static_cast<double>(x));
        return;
      }

      // Obtain the quantized output from model's output tensor
      int8_t no_quantized = output->data.int8[0];
      int8_t soft_quantized = output->data.int8[1];
      int8_t hard_quantized = output->data.int8[2];

      // Dequantize the output from integer to floating-point
      float no = (no_quantized - output->params.zero_point) * output->params.scale;
      float soft = (soft_quantized - output->params.zero_point) * output->params.scale;
      float hard = (hard_quantized - output->params.zero_point) * output->params.scale;

      /*
      Serial.print(no);
      Serial.print(',');
      Serial.print(soft);
      Serial.print(',');
      Serial.println(hard);
      */
      
      
      // Output the results. A custom HandleOutput function can be implemented
      // for each supported hardware target.
      //HandleOutput(error_reporter, x, y);


    }
}
