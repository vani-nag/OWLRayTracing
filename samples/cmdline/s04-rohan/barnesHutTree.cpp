#include "barnesHutTree.h"
#include <iostream>
#include <cmath>
#include <iostream>


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
}

BarnesHutTree::BarnesHutTree(float theta, float gridSize) : root(nullptr), theta(theta), gridSize(gridSize) {}

BarnesHutTree::~BarnesHutTree() {
  // todo free everything
}

void BarnesHutTree::insertNode(Node* node, const Point& point, float s) {
  int octant = 0;
  vec3float offset;
	offset.x = offset.y = offset.z = 0.0f;

  if(node->cofm.z < point.pos.z) {
    octant = 4;
    offset.z = s;
  }
  if(node->cofm.y < point.pos.y) {
    octant += 2;
    offset.y = s;
  }
  if(node->cofm.x < point.pos.x) {
    octant += 1;
    offset.x = s;
  }

  //printf("Octant is ->%d\n", octant);
  Node* child = node->children[octant];
  
  if(child == nullptr) {
    Node* new_node = new Node(point.pos.x, point.pos.y, point.pos.z, s, point.idX);
    new_node->mass = point.mass;
    node->children[octant] = new_node;
  } else {
    if(child->type == bhLeafNode) {
      // we need to split
      float half_r = 0.5 * s;
      Node* new_inner_node = new Node((node->cofm.x - half_r) + offset.x, (node->cofm.y - half_r) + offset.y, (node->cofm.z - half_r) + offset.z, half_r, -1);
      new_inner_node->type = bhNonLeafNode;

      BarnesHutTree::insertNode(new_inner_node, point, half_r);
      Point childPoint;
      childPoint.pos.x = child->cofm.x;
      childPoint.pos.y = child->cofm.y;
      childPoint.pos.z = child->cofm.z;
      childPoint.mass = child->mass;
      childPoint.idX = child->pointID;
      BarnesHutTree::insertNode(new_inner_node, childPoint, half_r);

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

// void BarnesHutTree::compute_center_of_mass(Node *root) {
// 	int i = 0;
// 	int j = 0;
// 	float mass;
//   vec3float cofm;
// 	vec3float cofm_child;
// 	Node* child;

// 	mass = 0.0;
// 	cofm.x = 0.0;
// 	cofm.y = 0.0; 
// 	cofm.z = 0.0;

// 	for (i = 0; i < 8; i++) {
// 		child = root->children[i];
// 		if (child != nullptr) {
// 			// compact child nodes for speed
// 			if (i != j) {
// 				root->children[j] = root->children[i];
// 				root->children[i] = 0;
// 			}

// 			j++;

// 			// If non leave node need to traverse children:
// 			if (child->type == bhNonLeafNode) {
// 				// summarize children
// 				compute_center_of_mass(child);
// 			} else {
// 				//*(points_sorted[sortidx++]) = child; // insert this point in sorted order
// 			}

// 			mass += child->mass;

// 			cofm_child.x = child->cofm.x * child->mass; // r*m
// 			cofm_child.y = child->cofm.y * child->mass; // r*m
// 			cofm_child.z = child->cofm.z * child->mass; // r*m

// 			cofm.x = cofm.x + cofm_child.x;
// 			cofm.y = cofm.y + cofm_child.y;
// 			cofm.z = cofm.z + cofm_child.z;
// 		}
// 	}

// 	cofm.x = cofm.x  * (1.0 / mass);
// 	cofm.y = cofm.y  * (1.0 / mass);
// 	cofm.z = cofm.z  * (1.0 / mass);

// 	root->cofm = cofm;
// 	root->mass = mass;
// }

void BarnesHutTree::printTree(Node* node, int depth = 0) {
  if(node == nullptr) {
    return;
  }

  // Print indentation based on depth
  for (int i = 0; i < depth; ++i) {
      std::cout << "  ";
  }

  std::cout << "└─ ";

  printf("Node: Mass = %f, Center of Mass = (%f, %f, %f)\n", node->mass, node->cofm.x, node->cofm.y, node->cofm.z);
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

float computeObjectsAttractionForce(Point point, Node *bhNode) {
  float mass_one = point.mass;
  float mass_two = bhNode->mass;

  // distance calculation
  float dx = point.pos.x - bhNode->cofm.x;
  float dy = point.pos.y - bhNode->cofm.y;
  float dz = point.pos.z - bhNode->cofm.z;
  float r_2 = (dx * dx) + (dy * dy) + (dz * dz);
  float result = (((mass_one * mass_two) / r_2) * GRAVITATIONAL_CONSTANT);

  //if(point.idX == 8124) printf("Distance is ->%.9f\n", result);
  return result;
}
 
float force_on(Point point, Node* node) {
  if(node->type == bhLeafNode) {
    //std::cout << "Node: Mass = " << node->mass << ", Center of Mass = (" << node->centerOfMassX << ", " << node->centerOfMassY << ")\n";
    if((node->mass != 0.0f) && ((point.pos.x != node->cofm.x) || (point.pos.y != node->cofm.y) || (point.pos.z != node->cofm.z))) {
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
    if(node->children[i] != nullptr) {
      totalForce += force_on(point, node->children[i]);
    }
  }

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