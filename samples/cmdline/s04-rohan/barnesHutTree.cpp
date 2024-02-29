#include "barnesHutTree.h"
#include <iostream>
#include <cmath>
#include <iostream>
#include <queue>


using namespace owl;

Node::Node(float x, float y, float z, float s, int pointID) {
  for(int i = 0; i < 8; i++) {
    children[i] = nullptr;
  }
  cofm.x = x;
  cofm.y = y;
  cofm.z = z;
  mass = 0.0f;
  this->s = s;
  quadrantX = 0.0f;
  quadrantY = 0.0f;
  quadrantZ = 0.0f;
  dfsIndex = 0;
  type = bhLeafNode;
  this->pointID = pointID;
  particles.reserve(BUCKET_SIZE);
}

BarnesHutTree::BarnesHutTree(float theta, float gridSize) : root(nullptr), theta(theta), gridSize(gridSize) {}

BarnesHutTree::~BarnesHutTree() {
  // todo free everything
}

// Traversal function to traverse the octree using Breadth-First Search (BFS)
void BarnesHutTree::traverseOctreeDFS(Node* node, std::vector<Node*>& leafNodes, float *minS) {
    if (node == nullptr) {
        return;
    }

    // If the node is a leaf node, add it to the array
    if (node->type == bhLeafNode) {
        if(node->s < *minS) {
          *minS = node->s;
        }
        leafNodes.push_back(node);
        return;
    }

    // Recursively traverse each child node
    for (int i = 0; i < 8; ++i) {
        if(node->children[i] != nullptr)
          BarnesHutTree::traverseOctreeDFS(node->children[i], leafNodes, minS);
    }
}

void BarnesHutTree::insertNode(Node* node, Node* point, float s) {
  int octant = 0;
  vec3float offset;
	offset.x = offset.y = offset.z = 0.0f;

  if(node->cofm.z < point->cofm.z) {
    octant = 4;
    offset.z = s;
  }
  if(node->cofm.y < point->cofm.y) {
    octant += 2;
    offset.y = s;
  }
  if(node->cofm.x < point->cofm.x) {
    octant += 1;
    offset.x = s;
  }

  //printf("Octant is ->%d\n", octant);
  Node* child = node->children[octant];
  
  if(child == nullptr) {
    // Node* new_node = new Node(point.pos.x, point.pos.y, point.pos.z, s, point.idX);
    // new_node->mass = point.mass;
    point->s = s;
    node->children[octant] = point;
  } else {
    if(child->type == bhLeafNode) {
      // we need to split
      float half_r = 0.5 * s;
      Node* new_inner_node = new Node((node->cofm.x - half_r) + offset.x, (node->cofm.y - half_r) + offset.y, (node->cofm.z - half_r) + offset.z, half_r, -1);
      new_inner_node->type = bhNonLeafNode;

      BarnesHutTree::insertNode(new_inner_node, point, half_r);
      BarnesHutTree::insertNode(new_inner_node, child, half_r);
			node->children[octant] = new_inner_node;
    } else {
      float half_r = 0.5 * s;
      BarnesHutTree::insertNode(child, point, half_r);
    }
  }
}

// Function to recursively compute the center of mass and total mass
void BarnesHutTree::computeCOM(Node* node) {
    if (node->type == bhNonLeafNode) {
        float totalMass = 0.0f;
        float cofm_x = 0.0f;
        float cofm_y = 0.0f;
        float cofm_z = 0.0f;

        for (int i = 0; i < 8; ++i) {
            if (node->children[i] != nullptr) {
                computeCOM(node->children[i]);
                totalMass += node->children[i]->mass;
                cofm_x += node->children[i]->cofm.x * node->children[i]->mass;
                cofm_y += node->children[i]->cofm.y * node->children[i]->mass;
                cofm_z += node->children[i]->cofm.z * node->children[i]->mass;
            }
        }

        if (totalMass != 0) {
            node->mass = totalMass;
            node->cofm.x = cofm_x / totalMass;
            node->cofm.y = cofm_y / totalMass;
            node->cofm.z = cofm_z / totalMass;
        }
    }
}

void BarnesHutTree::printTree(Node* node, int depth = 0) {
  if(node == nullptr) {
    return;
  }

  // Print indentation based on depth
  for (int i = 0; i < depth; ++i) {
      std::cout << "  ";
  }

  std::cout << "└─ ";

  printf("Node: Mass = %f, Center of Mass = (%f, %f, %f), Leaf = (%d)\n", node->mass, node->cofm.x, node->cofm.y, node->cofm.z, node->type);
  printf("( ");
  for(int i = 0; i < node->particles.size(); i++) {
    printf("%d ", node->particles[i]);
  }
  printf(")\n");
  for(int i = 0; i < 8; i++) {
    printTree(node->children[i], depth + 1);
  }
}

float distanceBetweenObjects(Point point, Node *bhNode) {
  // distance calculation
  float dx = point.pos.x - bhNode->cofm.x;
  float dy = point.pos.y - bhNode->cofm.y;
  float dz = point.pos.z - bhNode->cofm.z;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);

  return std::sqrt(r_2);
}

float computeObjectsAttractionForce(Point point, Node *bhNode, std::vector<Point> &points) {
  float result = 0.0f;
  if(bhNode->type == bhLeafNode) {
    for(int i = 0; i < bhNode->particles.size(); i++) {
      if(bhNode->particles[i] != point.idX) {
        float mass_one = point.mass;
        float mass_two = points[bhNode->particles[i]].mass;

        // distance calculation
        float dx = point.pos.x - points[bhNode->particles[i]].pos.x;
        float dy = point.pos.y - points[bhNode->particles[i]].pos.y;
        float dz = point.pos.z - points[bhNode->particles[i]].pos.z;
        float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
        result += (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);
      }
    } 
  } else {
    float mass_one = point.mass;
    float mass_two = bhNode->mass;

    // distance calculation
    float dx = point.pos.x - bhNode->cofm.x;
    float dy = point.pos.y - bhNode->cofm.y;
    float dz = point.pos.z - bhNode->cofm.z;
    float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
    result = (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);
  }

    //if(point.idX == 8124) printf("Distance is ->%.9f\n", result);
  return result;
}
 
float force_on(Point point, Node* node, std::vector<Point> &points) {
  if(node->type == bhLeafNode) {
    //std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << ")\n";
    if((node->mass != 0.0f) && ((point.pos.x != node->cofm.x) || (point.pos.y != node->cofm.y) || (point.pos.z != node->cofm.z))) {
      //if(point.idX == 5382) printf("Intersected leaf at node with mass! ->%f\n", node->mass);
      return computeObjectsAttractionForce(point, node, points);
    } else {
      return 0;
    }
  }

  if(node->s < distanceBetweenObjects(point, node) * THRESHOLD) {
    //if(point.idX == 0) printf("Approximate")
    //if(point.idX == 5382) printf("Approximated at node with mass! ->%f\n", node->mass);
    return computeObjectsAttractionForce(point, node, points);
  }

  float totalForce = 0;
  for(int i = 0; i < 8; i++) {
    if(node->children[i] != nullptr) {
      totalForce += force_on(point, node->children[i], points);
    }
  }

  return totalForce;
}

void BarnesHutTree::computeForces(Node* node, std::vector<Point> &points, std::vector<float>& cpuComputedForces) {
  for(int i = 0; i < points.size(); i++) {
    float force = 0;
    force = force_on(points[i], node, points);
    cpuComputedForces[i] = force;
    //printf("Point # %d has x = %f, y = %f, force = %f\n", i, points[i].x, points[i].y, force);
  }
}