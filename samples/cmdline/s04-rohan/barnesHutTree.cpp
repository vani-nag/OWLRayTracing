#include "barnesHutTree.h"
#include <iostream>
#include <cmath>
#include <iostream>


using namespace owl;

Node::Node(float x, float y, float z, float s) : quadrantX(x), quadrantY(y), quadrantZ(z), mass(0), s(s), centerOfMassX(0), centerOfMassY(0), centerOfMassZ(0) {
  for(int i = 0; i < 8; i++) {
    children[i] = nullptr;
  }
}

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
    node->centerOfMassZ = point.z;
    node->pointID = point.idX;
    return;
  }

  if(node->children[0] == nullptr) {
    splitNode(node);
  }

  // children order
  // 0 -> sw and back face
  // 1 -> nw and back face
  // 2 -> se and back face
  // 3 -> ne and back face
  // 4 -> sw and front face
  // 5 -> nw and front face
  // 6 -> se and front face
  // 7 -> ne and front face

  // determine quadrant to place point
  if(point.z < node->quadrantZ) {
    if(point.x < node->quadrantX) {
      if(point.y < node->quadrantY) {
        BarnesHutTree::insertNode(node->children[0], point);
      } else {
        BarnesHutTree::insertNode(node->children[1], point);
      }
    } else {
      if(point.y < node->quadrantY) {
        BarnesHutTree::insertNode(node->children[2], point);
      } else {
        BarnesHutTree::insertNode(node->children[3], point);
      }
    }
  } else {
    if(point.x < node->quadrantX) {
      if(point.y < node->quadrantY) {
        BarnesHutTree::insertNode(node->children[4], point);
      } else {
        BarnesHutTree::insertNode(node->children[5], point);
      }
    } else {
      if(point.y < node->quadrantY) {
        BarnesHutTree::insertNode(node->children[6], point);
      } else {
        BarnesHutTree::insertNode(node->children[7], point);
      }
    }
  }

  // update total mass and center of mass
  node->centerOfMassX = ((node->mass * node->centerOfMassX) + (point.mass * point.x)) / (node->mass + point.mass);
  node->centerOfMassY = ((node->mass * node->centerOfMassY) + (point.mass * point.y)) / (node->mass + point.mass);
  node->centerOfMassZ = ((node->mass * node->centerOfMassZ) + (point.mass * point.z)) / (node->mass + point.mass);
  node->mass += point.mass;
}

void BarnesHutTree::splitNode(Node* node) {
  float x = node->quadrantX;
  float y = node->quadrantY;
  float z = node->quadrantZ;
  float s = node->s / 2.0;
  float quadrantOperand = s / 2.0;

  // children order
  // 0 -> sw and back face
  // 1 -> nw and back face
  // 2 -> se and back face
  // 3 -> ne and back face
  // 4 -> sw and front face
  // 5 -> nw and front face
  // 6 -> se and front face
  // 7 -> ne and front face
  
  node->children[0] = new Node(x - quadrantOperand, y + quadrantOperand, z - quadrantOperand, s);
  node->children[1] = new Node(x - quadrantOperand, y - quadrantOperand, z - quadrantOperand, s);
  node->children[2] = new Node(x + quadrantOperand, y + quadrantOperand, z - quadrantOperand, s);
  node->children[3] = new Node(x + quadrantOperand, y - quadrantOperand, z - quadrantOperand, s);
  node->children[4] = new Node(x - quadrantOperand, y + quadrantOperand, z + quadrantOperand, s);
  node->children[5] = new Node(x - quadrantOperand, y - quadrantOperand, z + quadrantOperand, s);
  node->children[6] = new Node(x + quadrantOperand, y + quadrantOperand, z + quadrantOperand, s);
  node->children[7] = new Node(x + quadrantOperand, y - quadrantOperand, z + quadrantOperand, s);

  // node->nw = new Node(x - quadrantOperand, y + quadrantOperand, s);
  // node->ne = new Node(x + quadrantOperand, y + quadrantOperand, s);
  // node->sw = new Node(x - quadrantOperand, y - quadrantOperand, s);
  // node->se = new Node(x + quadrantOperand, y - quadrantOperand, s);

  if(node->centerOfMassZ < node->quadrantZ) {
    if(node->centerOfMassX < node->quadrantX) {
      if(node->centerOfMassY < node->quadrantY) {
        node->children[0]->mass = node->mass;
        node->children[0]->centerOfMassX = node->centerOfMassX;
        node->children[0]->centerOfMassY = node->centerOfMassY;
        node->children[0]->centerOfMassZ = node->centerOfMassZ;
        node->children[0]->pointID = node->pointID;
      } else {
        node->children[1]->mass = node->mass;
        node->children[1]->centerOfMassX = node->centerOfMassX;
        node->children[1]->centerOfMassY = node->centerOfMassY;
        node->children[1]->centerOfMassZ = node->centerOfMassZ;
        node->children[1]->pointID = node->pointID;
      }
    } else {
      if(node->centerOfMassY < node->quadrantY) {
        node->children[2]->mass = node->mass;
        node->children[2]->centerOfMassX = node->centerOfMassX;
        node->children[2]->centerOfMassY = node->centerOfMassY;
        node->children[2]->centerOfMassZ = node->centerOfMassZ;
        node->children[2]->pointID = node->pointID;
      } else {
        node->children[3]->mass = node->mass;
        node->children[3]->centerOfMassX = node->centerOfMassX;
        node->children[3]->centerOfMassY = node->centerOfMassY;
        node->children[3]->centerOfMassZ = node->centerOfMassZ;
        node->children[3]->pointID = node->pointID;
      }
    }
  } else {
    if(node->centerOfMassX < node->quadrantX) {
      if(node->centerOfMassY < node->quadrantY) {
        node->children[4]->mass = node->mass;
        node->children[4]->centerOfMassX = node->centerOfMassX;
        node->children[4]->centerOfMassY = node->centerOfMassY;
        node->children[4]->centerOfMassZ = node->centerOfMassZ;
        node->children[4]->pointID = node->pointID;
      } else {
        node->children[5]->mass = node->mass;
        node->children[5]->centerOfMassX = node->centerOfMassX;
        node->children[5]->centerOfMassY = node->centerOfMassY;
        node->children[5]->centerOfMassZ = node->centerOfMassZ;
        node->children[5]->pointID = node->pointID;
      }
    } else {
      if(node->centerOfMassY < node->quadrantY) {
        node->children[6]->mass = node->mass;
        node->children[6]->centerOfMassX = node->centerOfMassX;
        node->children[6]->centerOfMassY = node->centerOfMassY;
        node->children[6]->centerOfMassZ = node->centerOfMassZ;
        node->children[6]->pointID = node->pointID;
      } else {
        node->children[7]->mass = node->mass;
        node->children[7]->centerOfMassX = node->centerOfMassX;
        node->children[7]->centerOfMassY = node->centerOfMassY;
        node->children[7]->centerOfMassZ = node->centerOfMassZ;
        node->children[7]->pointID = node->pointID;
      }
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
    for(int i = 0; i < 8; i++) {
      printTree(node->children[i], depth + 1, "none");
    }
    // printTree(node->nw, depth + 1, "nw");
    // printTree(node->ne, depth + 1, "ne");
    // printTree(node->sw, depth + 1, "sw");
    // printTree(node->se, depth + 1, "se");
}

float distanceBetweenObjects(Point point, Node *bhNode) {
  // distance calculation
  float dx = point.x - bhNode->centerOfMassX;
  float dy = point.y - bhNode->centerOfMassY;
  float dz = point.z - bhNode->centerOfMassZ;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);

  return std::sqrt(r_2);
}

float computeObjectsAttractionForce(Point point, Node *bhNode) {
  float mass_one = point.mass;
  float mass_two = bhNode->mass;

  // distance calculation
  float dx = point.x - bhNode->centerOfMassX;
  float dy = point.y - bhNode->centerOfMassY;
  float dz = point.z - bhNode->centerOfMassZ;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
  float result = (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);

  //if(point.idX == 8124) printf("Distance is ->%.9f\n", result);
  return result;
}
 
float force_on(Point point, Node* node) {
  if(node->children[0] == nullptr) {
    //std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << ")\n";
    if((node->mass != 0.0f) && ((point.x != node->centerOfMassX) || (point.y != node->centerOfMassY) || (point.z != node->centerOfMassZ))) {
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
  for(int i = 0; i < 8; i++) {
    totalForce += force_on(point, node->children[i]);
  }
  // totalForce += force_on(point, node->nw);
  // totalForce += force_on(point, node->ne);
  // totalForce += force_on(point, node->sw);
  // totalForce += force_on(point, node->se);

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