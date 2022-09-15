#pragma once
#include <iostream>
class Object {
public:
  Object() : content(nullptr) {}
  template <typename ValueType>
  Object(const ValueType& value) : content(new Holder<ValueType>(value)) {}
  Object(const Object& other) : content(other.content ? other.content->clone() : nullptr) {}
  ~Object() { delete content; }

  template <typename ValueType>
  Object& operator=(const ValueType& rhs) {
	Object(rhs).swap(*this);
	return *this;
  }

  Object& operator=(Object rhs) {
	rhs.swap(*this);
	return *this;
  }

  Object& swap(Object& rhs) {
	std::swap(content, rhs.content);
	return *this;
  }

  template <typename ValueType>
  ValueType get() {
	return content ? static_cast<Holder<ValueType>*>(content)->held : nullptr;
  }

  const std::type_info& type() const {
	return content ? content->type() : typeid(void);
  }

  const bool isNull() const { return !content; }
  const bool nonNull() const { return content; }

private:
  class PlaceHolder {
  public:
	virtual ~PlaceHolder() {}
	virtual const std::type_info& type() const = 0;
	virtual PlaceHolder* clone() const = 0;
  };

  template <typename ValueType>
  class Holder : public PlaceHolder {
  public:
	explicit Holder(const ValueType& value) : held(value) {}
	virtual PlaceHolder* clone() const { return new Holder(held); }
	virtual const std::type_info& type() const { return typeid(ValueType); }
	ValueType held;
  };

  PlaceHolder* content;
};