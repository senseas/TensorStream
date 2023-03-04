#pragma once
#include <iostream>
#include <typeinfo>

#include "../core/None.h"
#include "../core/TensorConst.h"
#include "../lang/ForEach.h"
#include "../lang/Object.h"
#include "../lang/Tenser.h"

using namespace std;
namespace Objects {

Tenser<None*>* random(vector<int> shape) {
  Tenser<None*>* a = new Tenser<None*>(shape);
  ForEach::farEach(a, [](None** o) { *o = new None(); });
  return a;
}

Tenser<None*>* fill(vector<int> shape, double value, bool isGrad) {
  Tenser<None*>* a = new Tenser<None*>(shape);
  ForEach::farEach(a, [value, isGrad](None** o) { *o = new None(value, isGrad); });
  return a;
}

Tenser<None*>* zeroNones(vector<int> shape) {
  Tenser<None*>* a = new Tenser<None*>(shape);
  ForEach::farEach(a, [](None** o) { *o = new None(0, false); });
  return a;
}

Tenser<Tensor*>* zeroTensors(vector<int> shape) {
  Tenser<Tensor*>* a = new Tenser<Tensor*>(shape);
  ForEach::farEach(a, [](Tensor** o) { *o = new TensorConst(0); });
  return a;
}

template <typename T>
bool nonNull(T object) {
  return object != nullptr;
}

bool isFunction(Tensor* tensor) {
  return tensor->name == "Function";
}

bool isNone(Tensor* tensor) {
  std::size_t found = tensor->name.find("None");
  return found != std::string::npos;
}

vector<int> shapes(Tenser<Object*>* arr) {
  vector<int> list = arr->shape;
  vector<int> shape(arr->shape.size());
  Object* obj = arr->get<Object*>(shape);
  while (obj->nonNull()) {
    if (obj->type() == typeid(Tenser<Object*>*)) {
      Tenser<Object*>* a = obj->get<Tenser<Object*>*>();
      std::copy(a->shape.begin(), a->shape.end(), std::back_inserter(list));
      obj = a->get<Object*>(0);
    } else if (obj->type() == typeid(Tenser<None*>*)) {
      Tenser<None*>* a = obj->get<Tenser<None*>*>();
      std::copy(a->shape.begin(), a->shape.end(), std::back_inserter(list));
      break;
    } else if (obj->type() == typeid(Object*)) {
      obj = obj->get<Object*>();
    } else {
      break;
    }
  }
  return list;
}

Tenser<None*>* setPadding(Tenser<None*>* a, int padding) {
  if (padding == 0) return a;
  int height = a->shape[1], width = a->shape[0];
  Tenser<None*>* nones = new Tenser<None*>({height + 2 * padding, width + 2 * padding});

  ForEach::forEach(padding, nones->shape[0],[nones, padding, height](int m, int n) {
		None* o = nones->get<None*>(m, n);
		None* p = new None(0, false);

		None* e = nones->get<None*>(m + padding + height, n);
		e = new None(0, false);
	});

  ForEach::forEach(nones->shape[1], padding, [nones, padding, width](int m, int n) {
		None* e = nones->get<None*>(m, n);
		e = new None(0, false);

		None* p = nones->get<None*>(m, n + padding + width);
		p = new None(0, false);
	});

  ForEach::forEach(height, width, [a, nones, padding](int i, int l) {
    None* c = nones->getx<None*>(i + padding, l + padding);

    c = a->get<None*>(i, l);
  });
  return nones;
}
}  // namespace Objects