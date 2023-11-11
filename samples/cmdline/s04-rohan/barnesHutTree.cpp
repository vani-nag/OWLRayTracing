#include "barnesHutTree.h"
#include <iostream>
#include <cmath>
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
  float quadrantOperand = s / 2.0;
  
  node->nw = new Node(x - quadrantOperand, y + quadrantOperand, s);
  node->ne = new Node(x + quadrantOperand, y + quadrantOperand, s);
  node->sw = new Node(x - quadrantOperand, y - quadrantOperand, s);
  node->se = new Node(x + quadrantOperand, y - quadrantOperand, s);

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

void BarnesHutTree::printTree(Node* node, int depth = 0, std::string corner = "none") {
    if (node == nullptr) {
        return;
    }

    // Print indentation based on depth
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }

    std::cout << "└─ ";

    // Print node information
    std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << "), quadrant = (" << node->quadrantX << ", " << node->quadrantY << "), corner = " << corner << "\n";

    // Recursively print child nodes
    printTree(node->nw, depth + 1, "nw");
    printTree(node->ne, depth + 1, "ne");
    printTree(node->sw, depth + 1, "sw");
    printTree(node->se, depth + 1, "se");
}

float distanceBetweenObjects(Point point, Node *bhNode) {
  // distance calculation
  float dx = point.x - bhNode->centerOfMassX;
  float dy = point.y - bhNode->centerOfMassY;
  float r_2 = (dx * dx) + (dy * dy);

  return std::sqrt(r_2);
}

float computeObjectsAttractionForce(Point point, Node *bhNode) {
  float mass_one = point.mass;
  float mass_two = bhNode->mass;

  // distance calculation
  float dx = point.x - bhNode->centerOfMassX;
  float dy = point.y - bhNode->centerOfMassY;
  float r_2 = (dx * dx) + (dy * dy);
  float result = (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);

  //if(point.idX == 8124) printf("Distance is ->%.9f\n", result);
  return result;
}
 
float force_on(Point point, Node* node) {
  if(node->nw == nullptr) {
    //std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << ")\n";
    if((node->mass != 0.0f) && ((point.x != node->centerOfMassX) || (point.y != node->centerOfMassY))) {
      //if(point.idX == 5382) printf("Intersected leaf at node with mass! ->%f\n", node->mass);
      return computeObjectsAttractionForce(point, node);
    } else {
      return 0;
    }
  }

  if(node->s < distanceBetweenObjects(point, node) * THRESHOLD) {
    //if(point.idX == 0) printf("Approximate")
    //if(point.idX == 5382) printf("Approximated at node with mass! ->%f\n", node->mass);
    return computeObjectsAttractionForce(point, node);
  }

  float totalForce = 0;
  totalForce += force_on(point, node->nw);
  totalForce += force_on(point, node->ne);
  totalForce += force_on(point, node->sw);
  totalForce += force_on(point, node->se);

  return totalForce;
}

void BarnesHutTree::computeForces(Node* node, std::vector<Point> points, std::vector<float>& cpuComputedForces) {
  for(int i = 0; i < points.size(); i++) {
    float force = 0;
    force = force_on(points[i], node);
    cpuComputedForces[i] = force;
    //printf("Point # %d has x = %f, y = %f, force = %f\n", i, points[i].x, points[i].y, force);
  }
}