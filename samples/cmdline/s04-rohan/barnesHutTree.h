#pragma once

#include <owl/owl.h>
#include <string>
#include <vector>

#define GRID_SIZE 10.0f
#define THRESHOLD 0.5f
#define GRAVITATIONAL_CONSTANT 1.0f

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
    bool isLeaf;

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
      isLeaf = false;
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
      void printTree(Node* root, int depth, std::string corner);
      void computeForces(Node* node, std::vector<Point> points, std::vector<float>& cpuComputedForces);
      //void calculateCenterOfMass();
  };
}




