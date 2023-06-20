#include "bitmap.h"
#include <algorithm>
#include <cstddef>

char *createBitmap(unsigned long numElements, bool initToZero = true) {
  char *bitmap;
  size_t bitmapSize = numElements / 8;
  if (numElements % 8 != 0) {
    bitmapSize++;
  }
  if(initToZero) {
    bitmap = new char[bitmapSize]();
  } else {
    bitmap = new char[bitmapSize];
    fill_n(bitmap, bitmapSize, 0xFF);
  }
  return bitmap;
}

char getBitAtPositionInBitmap(char *bitmap, unsigned long position) {
  unsigned long bytePosition = position / 8;
  unsigned long bitPosition = position % 8;

  char byte = bitmap[bytePosition];
  char bit = (byte >> bitPosition) & 1;
  return bit;
}

void setBitAtPositionInBitmap(char *bitmap, unsigned long position, char value) {
  unsigned long bytePosition = position / 8;
  unsigned long bitPosition = position % 8;

  char byte = bitmap[bytePosition];
  char bit = (byte >> bitPosition) & 1;
  if(bit != value) {
    bitmap[bytePosition] ^= (1 << bitPosition);
  }
}

size_t getBitmapSizeGivenNumberOfElements(size_t numElements) {
  size_t bitmapSize = numElements / 8;
  if (numElements % 8 != 0) {
    bitmapSize++;
  }
  return bitmapSize;
}