#pragma once

#include <stdbool.h>
#include <cstddef>

using namespace std;

char *createBitmap(unsigned long numElements, bool initToZero);
char getBitAtPositionInBitmap(char *bitmap, unsigned long position);
void setBitAtPositionInBitmap(char *bitmap, unsigned long position, char value);
size_t getBitmapSizeGivenNumberOfElements(size_t numElements);