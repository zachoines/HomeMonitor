#pragma once
namespace HM {
	class Servo
	{
	private:
		int currentAngle;

	public:
		Servo(int pin);
		~Servo();
		void setAngle(float angle);
	};
};

