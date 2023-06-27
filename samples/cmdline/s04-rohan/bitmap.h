#pragma once


#include <stdbool.h>
#include <cstddef>
#include <cstdint>

using namespace std;

uint32_t *createBitmap(unsigned long numElements, bool initToZero);
uint32_t getBitAtPositionInBitmap(uint32_t *bitmap, unsigned long position);
void setBitAtPositionInBitmap(uint32_t *bitmap, unsigned long position, uint32_t value);
size_t getBitmapSizeGivenNumberOfElements(size_t numElements);
unsigned long getNumberOfElementsGivenBitmapSize(size_t bitmapSize);
