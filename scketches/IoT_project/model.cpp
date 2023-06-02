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

// Automatically created from a TensorFlow Lite flatbuffer using the command:
// xxd -i model.tflite > model.cc

// This is a standard TensorFlow Lite model file that has been converted into a
// C data array, so it can be easily compiled into a binary for devices that
// don't have a file system.

// See train/README.md for a full description of the creation process.

#include "model.h"

// Keep model aligned to 8 bytes to guarantee aligned 64-bit accesses.
alignas(8) const unsigned char g_model[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x98, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0xe4, 0x05, 0x00, 0x00,
  0xf4, 0x05, 0x00, 0x00, 0x94, 0x0b, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f,
  0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x90, 0xff, 0xff, 0xff, 0x07, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x64, 0x65, 0x6e, 0x73,
  0x65, 0x5f, 0x31, 0x35, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x06, 0xfa, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x34,
  0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x34, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xdc, 0xff, 0xff, 0xff,
  0x0a, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x43, 0x4f, 0x4e, 0x56, 0x45, 0x52, 0x53, 0x49, 0x4f, 0x4e, 0x5f, 0x4d,
  0x45, 0x54, 0x41, 0x44, 0x41, 0x54, 0x41, 0x00, 0x08, 0x00, 0x0c, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x6e, 0x5f,
  0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x76, 0x65, 0x72, 0x73,
  0x69, 0x6f, 0x6e, 0x00, 0x0b, 0x00, 0x00, 0x00, 0xf0, 0x04, 0x00, 0x00,
  0xe8, 0x04, 0x00, 0x00, 0xcc, 0x04, 0x00, 0x00, 0xa4, 0x04, 0x00, 0x00,
  0x74, 0x04, 0x00, 0x00, 0xa4, 0x00, 0x00, 0x00, 0x9c, 0x00, 0x00, 0x00,
  0x94, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xae, 0xfa, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00,
  0x58, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0e, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0xeb, 0x03, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x10, 0x00, 0x0c, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x32, 0x2e, 0x31, 0x32, 0x2e, 0x30, 0x00, 0x00, 0x12, 0xfb, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x31, 0x2e, 0x31, 0x34,
  0x2e, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfc, 0xfa, 0xff, 0xff, 0x00, 0xfb, 0xff, 0xff, 0x04, 0xfb, 0xff, 0xff,
  0x3a, 0xfb, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0xc0, 0x03, 0x00, 0x00,
  0xbe, 0xf8, 0x01, 0xff, 0x08, 0x05, 0x90, 0x06, 0x01, 0x01, 0x0f, 0xff,
  0xb6, 0xe9, 0x03, 0x03, 0xfa, 0x0c, 0xd1, 0xf7, 0x02, 0xfd, 0xf8, 0x09,
  0xbb, 0xf4, 0x04, 0xfd, 0x04, 0xf9, 0xb0, 0xe9, 0xff, 0x00, 0x0b, 0xfb,
  0xd9, 0xec, 0xfd, 0xfe, 0x00, 0x04, 0xcc, 0xf0, 0x02, 0x04, 0xf7, 0x0a,
  0xc3, 0xed, 0x00, 0x01, 0x03, 0x06, 0xd1, 0xec, 0x02, 0x01, 0x02, 0x04,
  0xd3, 0xee, 0x01, 0xff, 0xf9, 0x04, 0xd9, 0xf0, 0xfd, 0x00, 0xf7, 0x08,
  0xbe, 0xfc, 0x01, 0xfa, 0xfe, 0x09, 0xc7, 0x10, 0x03, 0x03, 0x07, 0x08,
  0xbd, 0x09, 0xfd, 0x02, 0x03, 0x0d, 0xad, 0x01, 0xfe, 0x05, 0x00, 0x0d,
  0x9f, 0xf7, 0xff, 0x03, 0xfb, 0x09, 0x98, 0xfe, 0x01, 0x05, 0xfb, 0x05,
  0x8d, 0x00, 0x04, 0xfa, 0x01, 0x04, 0x81, 0xfa, 0x02, 0xf7, 0x04, 0x04,
  0x2e, 0xee, 0x02, 0x04, 0x04, 0xff, 0x50, 0xf1, 0x04, 0xf8, 0x04, 0xfc,
  0x3c, 0x00, 0x02, 0xfa, 0x02, 0x00, 0x2f, 0xf2, 0x02, 0xf7, 0x05, 0xfe,
  0x41, 0xfd, 0x04, 0xfb, 0x04, 0xf9, 0x3e, 0x03, 0x07, 0xfa, 0x01, 0xf8,
  0x2d, 0xfe, 0x03, 0xf6, 0x07, 0xf7, 0x35, 0xfe, 0x03, 0xf6, 0x05, 0xff,
  0x30, 0x02, 0x04, 0xfb, 0x02, 0xfd, 0x32, 0xfb, 0xff, 0xfb, 0x09, 0xff,
  0x36, 0xf6, 0x07, 0xf9, 0x04, 0xfa, 0x2e, 0xfb, 0x07, 0xf7, 0x02, 0xff,
  0x44, 0xf4, 0x07, 0xf6, 0x04, 0xfa, 0x43, 0xe6, 0x03, 0xf9, 0x07, 0xfb,
  0x4b, 0xe7, 0x00, 0xf6, 0x06, 0xf5, 0x3e, 0xf5, 0x00, 0xfa, 0x00, 0xfb,
  0x47, 0xf4, 0x07, 0xf9, 0x06, 0xf8, 0x6a, 0xf1, 0x03, 0xf9, 0x07, 0xf6,
  0x68, 0xee, 0x06, 0xf8, 0x05, 0xf2, 0x60, 0xf2, 0x00, 0xfd, 0x00, 0xef,
  0x26, 0x1e, 0x05, 0x00, 0xf9, 0xfd, 0x48, 0x21, 0x06, 0xfd, 0xfc, 0xfe,
  0x3b, 0x31, 0x03, 0x05, 0x01, 0x07, 0x25, 0x28, 0x05, 0x00, 0x05, 0x04,
  0x35, 0x2f, 0x00, 0x07, 0xfc, 0x04, 0x2c, 0x38, 0x07, 0x02, 0xff, 0x04,
  0x1e, 0x36, 0x05, 0x04, 0xfe, 0xfc, 0x1f, 0x2f, 0x03, 0x07, 0xfd, 0x03,
  0x1e, 0x34, 0xff, 0x05, 0x02, 0x06, 0x16, 0x34, 0x05, 0x08, 0x07, 0x06,
  0x1f, 0x2a, 0x02, 0x06, 0x06, 0x06, 0x11, 0x31, 0x06, 0x02, 0x00, 0x0b,
  0x20, 0x2c, 0xfe, 0x01, 0x00, 0x0c, 0x2a, 0x1a, 0x01, 0x00, 0x00, 0x0b,
  0x2f, 0x1f, 0x05, 0x03, 0x01, 0x08, 0x28, 0x27, 0x06, 0x0a, 0xff, 0x06,
  0x38, 0x27, 0x03, 0x07, 0xfd, 0x09, 0x58, 0x23, 0x00, 0x0e, 0x00, 0x06,
  0x5b, 0x29, 0x01, 0x02, 0xfe, 0xff, 0x4d, 0x2c, 0xff, 0x02, 0x03, 0xfa,
  0xc5, 0xf9, 0x03, 0x03, 0xfb, 0x0b, 0xb2, 0xfe, 0x00, 0xff, 0xf9, 0x0a,
  0xb9, 0xe7, 0x04, 0xff, 0x06, 0xfa, 0xdb, 0xf6, 0x05, 0x04, 0x01, 0x04,
  0xbf, 0xed, 0x05, 0x01, 0xfc, 0x07, 0xbc, 0xe9, 0x04, 0xfe, 0x01, 0x04,
  0xd8, 0xe6, 0xfd, 0x03, 0x0b, 0xfe, 0xdb, 0xea, 0x03, 0xfb, 0xf8, 0xfe,
  0xc8, 0xe6, 0x03, 0xfe, 0xf6, 0x01, 0xde, 0xec, 0x00, 0xfb, 0x00, 0x05,
  0xb8, 0xf8, 0xfe, 0x00, 0x07, 0x07, 0xde, 0xef, 0xfd, 0xff, 0x01, 0x07,
  0xbd, 0xf8, 0x04, 0x07, 0xfa, 0x08, 0xb6, 0x08, 0xfd, 0xfe, 0xfa, 0x00,
  0xac, 0x02, 0xfd, 0xfd, 0x00, 0x00, 0xbb, 0x00, 0x01, 0xf9, 0x02, 0x05,
  0xaa, 0xf5, 0x04, 0xfe, 0xfa, 0x01, 0x8d, 0xfe, 0xff, 0xfe, 0x01, 0xff,
  0x8c, 0xff, 0x03, 0x03, 0x02, 0xfe, 0x91, 0xfa, 0x04, 0x05, 0xfd, 0x04,
  0xbc, 0xfd, 0x02, 0x00, 0xf9, 0xfd, 0xd0, 0xf5, 0x03, 0x01, 0xfb, 0x01,
  0xb1, 0xf8, 0x02, 0xff, 0xfd, 0x03, 0x96, 0xf4, 0x01, 0xff, 0x0e, 0xfc,
  0xcb, 0xf4, 0x02, 0x01, 0xfd, 0x07, 0xf5, 0xea, 0x02, 0xff, 0xf4, 0x03,
  0xcb, 0xf5, 0xff, 0xff, 0xf9, 0xfe, 0xcf, 0xef, 0x04, 0x00, 0x0b, 0x01,
  0xc3, 0xed, 0xfd, 0x04, 0xfe, 0x09, 0xbe, 0xed, 0x03, 0x02, 0xff, 0x04,
  0xc1, 0xfe, 0xff, 0x01, 0x01, 0x01, 0xd1, 0xef, 0xfe, 0xfd, 0x03, 0x04,
  0xdb, 0x05, 0x02, 0xfb, 0x03, 0x04, 0xc8, 0x0d, 0xfd, 0xfa, 0xff, 0x04,
  0xca, 0xfc, 0x03, 0xfe, 0xfb, 0xff, 0xcc, 0xf0, 0x04, 0x03, 0xf7, 0xf9,
  0xc2, 0x00, 0x00, 0x03, 0x06, 0xfd, 0xa3, 0x05, 0x04, 0x04, 0x04, 0x00,
  0x94, 0xfe, 0xfd, 0x03, 0xfc, 0x04, 0x9c, 0xfa, 0x02, 0x03, 0xfd, 0x04,
  0xd8, 0x0d, 0xff, 0xfe, 0x08, 0xfd, 0xb3, 0x0a, 0x00, 0x06, 0x04, 0x01,
  0xbf, 0xf9, 0x02, 0x02, 0x02, 0x00, 0xce, 0x0b, 0x04, 0x02, 0xfc, 0x07,
  0xbf, 0x03, 0xfe, 0x01, 0x03, 0x05, 0xc3, 0xf7, 0x05, 0x03, 0x05, 0x09,
  0xd8, 0xf9, 0x05, 0x05, 0x00, 0x10, 0xd0, 0xfc, 0x05, 0x03, 0x02, 0x09,
  0xd1, 0xf6, 0x03, 0xfd, 0x00, 0x05, 0xd2, 0xfb, 0x05, 0x03, 0xfa, 0x08,
  0xce, 0x07, 0xff, 0x00, 0xfa, 0x09, 0xdc, 0x02, 0x04, 0x02, 0xff, 0x06,
  0xc4, 0x08, 0x01, 0x06, 0x07, 0x08, 0xbf, 0x1b, 0xff, 0x07, 0x01, 0x07,
  0xb9, 0x13, 0x03, 0x06, 0xfc, 0x09, 0xc1, 0x0d, 0x04, 0x01, 0x01, 0x0a,
  0xb8, 0x0a, 0x03, 0x01, 0xff, 0x0b, 0x9d, 0x12, 0x03, 0x00, 0x02, 0x05,
  0x99, 0x0b, 0x02, 0x02, 0x00, 0x0d, 0xa3, 0x0a, 0x01, 0x02, 0xff, 0x0a,
  0x2c, 0x1e, 0x01, 0xfd, 0xf9, 0xfd, 0x54, 0x18, 0x02, 0x04, 0x01, 0xf7,
  0x44, 0x28, 0x05, 0x07, 0xf9, 0x02, 0x31, 0x21, 0x02, 0x01, 0x04, 0xfe,
  0x42, 0x24, 0x05, 0x07, 0xff, 0xf9, 0x37, 0x34, 0x05, 0x04, 0xfe, 0xf8,
  0x2a, 0x2b, 0x06, 0x03, 0x00, 0xf5, 0x31, 0x28, 0x05, 0x0a, 0x03, 0xf7,
  0x2c, 0x32, 0x06, 0x07, 0xfd, 0xfb, 0x29, 0x2c, 0x03, 0x03, 0x09, 0xf8,
  0x30, 0x24, 0x03, 0x07, 0x05, 0xfc, 0x21, 0x28, 0x05, 0x03, 0xfc, 0xfd,
  0x38, 0x24, 0x03, 0x07, 0x01, 0xff, 0x3b, 0x15, 0x00, 0x05, 0x03, 0xfd,
  0x49, 0x19, 0x01, 0x03, 0x03, 0xfb, 0x3a, 0x1d, 0x06, 0x08, 0xfc, 0xfd,
  0x42, 0x22, 0x02, 0x06, 0x02, 0xf6, 0x6a, 0x1b, 0xff, 0x07, 0x03, 0xfd,
  0x69, 0x1f, 0x04, 0x0a, 0x00, 0xfe, 0x64, 0x21, 0x05, 0x06, 0xfa, 0x04,
  0xd3, 0x17, 0x03, 0x00, 0x02, 0xfe, 0xb6, 0x15, 0x03, 0x04, 0x02, 0x03,
  0xc1, 0x05, 0x00, 0x07, 0x05, 0x03, 0xd5, 0x16, 0x01, 0x04, 0x00, 0x02,
  0xc7, 0x12, 0x05, 0x03, 0x03, 0x06, 0xc5, 0x07, 0x05, 0x07, 0x01, 0x09,
  0xd4, 0x05, 0xfe, 0x02, 0x00, 0x0c, 0xd2, 0x13, 0x06, 0x02, 0x04, 0x07,
  0xd0, 0x0d, 0xff, 0x07, 0x02, 0x07, 0xd3, 0x0c, 0x06, 0x04, 0x00, 0x05,
  0xc8, 0x11, 0x01, 0x05, 0x00, 0x07, 0xdb, 0x12, 0x00, 0x06, 0xff, 0x07,
  0xc1, 0x15, 0x05, 0x08, 0x00, 0x08, 0xc3, 0x23, 0x06, 0x04, 0x01, 0x07,
  0xb8, 0x20, 0x01, 0x06, 0x00, 0x05, 0xbe, 0x1a, 0x02, 0x00, 0x02, 0x09,
  0xb6, 0x18, 0x00, 0x0a, 0xfd, 0x0f, 0xa0, 0x1e, 0x01, 0x03, 0x00, 0x0b,
  0x9e, 0x1d, 0x06, 0x00, 0x02, 0x05, 0xa8, 0x1d, 0x01, 0x05, 0xfe, 0x0b,
  0x06, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x32, 0xff, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0xe8, 0x24, 0xb0, 0x01,
  0xd6, 0x81, 0xf4, 0x94, 0x4d, 0xf6, 0x05, 0x4d, 0x79, 0x2b, 0xf4, 0x1d,
  0xa3, 0x07, 0x27, 0xa1, 0x8f, 0xf1, 0x18, 0x05, 0x56, 0xff, 0xff, 0xff,
  0x04, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0xf8, 0xff, 0xff, 0xff, 0x06, 0x00, 0x00, 0x00, 0x3c, 0xff, 0xff, 0xff,
  0x40, 0xff, 0xff, 0xff, 0x0f, 0x00, 0x00, 0x00, 0x4d, 0x4c, 0x49, 0x52,
  0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0xf0, 0x00, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x1a, 0x00, 0x14, 0x00,
  0x10, 0x00, 0x0c, 0x00, 0x0b, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x1c, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0xca, 0xff, 0xff, 0xff, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x08, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x0b, 0x00, 0x04, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
  0x18, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0xec, 0x03, 0x00, 0x00,
  0x54, 0x03, 0x00, 0x00, 0xc8, 0x02, 0x00, 0x00, 0x4c, 0x02, 0x00, 0x00,
  0xd8, 0x01, 0x00, 0x00, 0x1c, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x4a, 0xfc, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x50, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00,
  0x34, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3b, 0x19, 0x00, 0x00, 0x00,
  0x53, 0x74, 0x61, 0x74, 0x65, 0x66, 0x75, 0x6c, 0x50, 0x61, 0x72, 0x74,
  0x69, 0x74, 0x69, 0x6f, 0x6e, 0x65, 0x64, 0x43, 0x61, 0x6c, 0x6c, 0x3a,
  0x30, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0xc2, 0xfc, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x70, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x00, 0x00,
  0xac, 0xfc, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x9a, 0xf2, 0x1b, 0x40, 0x3a, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x35, 0x2f, 0x4d, 0x61,
  0x74, 0x4d, 0x75, 0x6c, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74,
  0x69, 0x61, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f,
  0x31, 0x35, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x5a, 0xfd, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x90, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0xff, 0xff, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x44, 0xfd, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xcd, 0x60, 0xfb, 0x3e, 0x55, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x34, 0x2f, 0x4d, 0x61,
  0x74, 0x4d, 0x75, 0x6c, 0x3b, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74,
  0x69, 0x61, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f,
  0x31, 0x34, 0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e,
  0x73, 0x65, 0x5f, 0x31, 0x34, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64,
  0x64, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0xa6, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
  0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x09, 0x4c, 0x00, 0x00, 0x00, 0xec, 0xfd, 0xff, 0xff,
  0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xe4, 0xde, 0x57, 0x3d, 0x1c, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x34, 0x2f, 0x4d, 0x61,
  0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x16, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x58, 0x00, 0x00, 0x00,
  0x5c, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xa7, 0xa5, 0x93, 0x3d, 0x2c, 0x00, 0x00, 0x00,
  0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35,
  0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31, 0x34, 0x2f, 0x42, 0x69,
  0x61, 0x73, 0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61,
  0x72, 0x69, 0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x8e, 0xff, 0xff, 0xff,
  0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x4c, 0x00, 0x00, 0x00,
  0xd4, 0xfe, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc9, 0xef, 0x09, 0x3d,
  0x1c, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x74, 0x69,
  0x61, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x31,
  0x35, 0x2f, 0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x16, 0x00, 0x1c, 0x00, 0x18, 0x00, 0x17, 0x00, 0x10, 0x00,
  0x0c, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x00,
  0x34, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
  0x5c, 0x00, 0x00, 0x00, 0x5c, 0xff, 0xff, 0xff, 0x08, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x42, 0x72, 0x87, 0x3c, 0x2c, 0x00, 0x00, 0x00, 0x73, 0x65, 0x71, 0x75,
  0x65, 0x6e, 0x74, 0x69, 0x61, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e,
  0x73, 0x65, 0x5f, 0x31, 0x35, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64,
  0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62,
  0x6c, 0x65, 0x4f, 0x70, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x20, 0x00, 0x1c, 0x00,
  0x1b, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x18, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x68, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x78, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x18, 0xaf, 0x3f,
  0x20, 0x00, 0x00, 0x00, 0x73, 0x65, 0x72, 0x76, 0x69, 0x6e, 0x67, 0x5f,
  0x64, 0x65, 0x66, 0x61, 0x75, 0x6c, 0x74, 0x5f, 0x64, 0x65, 0x6e, 0x73,
  0x65, 0x5f, 0x31, 0x34, 0x5f, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x3a, 0x30,
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x78, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0xf0, 0xff, 0xff, 0xff, 0x19, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x0c, 0x00, 0x10, 0x00,
  0x0f, 0x00, 0x00, 0x00, 0x08, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09};
const int g_model_len = 2488;
