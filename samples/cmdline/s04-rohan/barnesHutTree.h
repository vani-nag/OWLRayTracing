#pragma once

#include <owl/owl.h>

namespace owl {

  struct Point {
    float x;
    float y;
    float mass;
    int idX;
  };


  struct Node {
    float quadrantX;
    float quadrantY;
    float mass;
    float s;
    float centerOfMassX;
    float centerOfMassY;
    Node* nw;
    Node* ne;
    Node* sw;
    Node* se;

    Node(float x, float y, float s);

    Node() {
      mass = 0;
      s = 0;
      centerOfMassX = 0;
      centerOfMassY = 0;
      nw = nullptr;
      ne = nullptr;
      sw = nullptr;
      se = nullptr;
    }
  };

  class BarnesHutTree {
    private:
      Node* root;
      float theta;
      float gridSize;

      void splitNode(Node* node);
      //void calculateCenterOfMass(Node* node);

    public:
      BarnesHutTree(float theta, float gridSize);
      ~BarnesHutTree();

      void insertNode(Node* node, const Point& point);
      void printTree(Node* root, int depth);
      //void calculateCenterOfMass();
  };
}




