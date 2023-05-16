#include "barnesHutTree.h"
#include <iostream>

using namespace owl;

Node::Node(float x, float y, float s) : quadrantX(x), quadrantY(y), mass(0), s(s), centerOfMassX(0), centerOfMassY(0), nw(nullptr), ne(nullptr), sw(nullptr), se(nullptr) {}

BarnesHutTree::BarnesHutTree(float theta, float gridSize) : root(nullptr), theta(theta), gridSize(gridSize) {}

BarnesHutTree::~BarnesHutTree() {
  // todo free everything
}

void BarnesHutTree::insertNode(Node* node, const Point& point) {

  // base case
  if(node->mass == 0) {
    node->mass = point.mass;
    node->centerOfMassX = point.x;
    node->centerOfMassY = point.y;
    return;
  }

  if(node->nw == nullptr) {
    splitNode(node);
  }

  // determine quadrant to place point
  if(point.x < node->quadrantX) {
    if(point.y < node->quadrantY) {
      BarnesHutTree::insertNode(node->sw, point);
    } else {
      BarnesHutTree::insertNode(node->nw, point);
    }
  } else {
    if(point.y < node->quadrantY) {
      BarnesHutTree::insertNode(node->se, point);
    } else {
      BarnesHutTree::insertNode(node->ne, point);
    }
  }

  // update total mass and center of mass
  node->centerOfMassX = ((node->mass * node->centerOfMassX) + (point.mass * point.x)) / (node->mass + point.mass);
  node->centerOfMassY = ((node->mass * node->centerOfMassY) + (point.mass * point.y)) / (node->mass + point.mass);
  node->mass += point.mass;
}

void BarnesHutTree::splitNode(Node* node) {
  float x = node->quadrantX;
  float y = node->quadrantY;
  float s = node->s / 2.0;
  
  node->nw = new Node(x - s, y + s, s);
  node->ne = new Node(x + s, y + s, s);
  node->sw = new Node(x - s, y - s, s);
  node->se = new Node(x + s, y - s, s);

  if(node->centerOfMassX < node->quadrantX) {
    if(node->centerOfMassY < node->quadrantY) {
      node->sw->mass = node->mass;
      node->sw->centerOfMassX = node->centerOfMassX;
      node->sw->centerOfMassY = node->centerOfMassY;
    } else {
      node->nw->mass = node->mass;
      node->nw->centerOfMassX = node->centerOfMassX;
      node->nw->centerOfMassY = node->centerOfMassY;
    }
  } else {
    if(node->centerOfMassY < node->quadrantY) {
      node->se->mass = node->mass;
      node->se->centerOfMassX = node->centerOfMassX;
      node->se->centerOfMassY = node->centerOfMassY;
    } else {
      node->ne->mass = node->mass;
      node->ne->centerOfMassX = node->centerOfMassX;
      node->ne->centerOfMassY = node->centerOfMassY;
    }
  }
}

void BarnesHutTree::printTree(Node* node, int depth = 0) {
    if (node == nullptr) {
        return;
    }

    // Print indentation based on depth
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }

    std::cout << "└─ ";

    // Print node information
    std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << ")\n";

    // Recursively print child nodes
    printTree(node->nw, depth + 1);
    printTree(node->ne, depth + 1);
    printTree(node->se, depth + 1);
    printTree(node->sw, depth + 1);
}