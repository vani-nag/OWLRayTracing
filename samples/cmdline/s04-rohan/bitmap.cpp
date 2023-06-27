#include "bitmap.h"
#include <algorithm>
#include <cstddef>

u_int *createBitmap(unsigned long numElements, bool initToZero = true) {
  u_int *bitmap;
  size_t bitmapSize = numElements / 32;
  if (numElements % 32 != 0) {
    bitmapSize++;
  }
  if(initToZero) {
    bitmap = new u_int[bitmapSize]();
  } else {
    bitmap = new u_int[bitmapSize];
    fill_n(bitmap, bitmapSize, 0xFFFFFFFF);
  }
  return bitmap;
}

u_int getBitAtPositionInBitmap(u_int *bitmap, unsigned long position) {
  unsigned long bytePosition = position / 32;
  unsigned long bitPosition = position % 32;

  u_int byte = bitmap[bytePosition];
  u_int bit = (byte >> bitPosition) & 1;
  return bit;
}

void setBitAtPositionInBitmap(u_int *bitmap, unsigned long position, u_int value) {
  unsigned long bytePosition = position / 32;
  unsigned long bitPosition = position % 32;

  u_int byte = bitmap[bytePosition];
  u_int bit = (byte >> bitPosition) & 1;
  if(bit != value) {
    bitmap[bytePosition] ^= (1 << bitPosition);
  }
}

size_t getBitmapSizeGivenNumberOfElements(size_t numElements) {
  size_t bitmapSize = numElements / 32;
  if (numElements % 32 != 0) {
    bitmapSize++;
  }
  return bitmapSize;
}

unsigned long getNumberOfElementsGivenBitmapSize(size_t bitmapSize) {
  return bitmapSize / sizeof(u_int);
}