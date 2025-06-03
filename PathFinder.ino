#include "Adafruit_VL53L0X.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
// s0=11; s1=10; s2=12; s3=13; s4=14; s5=15;
/*
Adafruit_VL53L0X l = Adafruit_VL53L0X();
Adafruit_VL53L0X l45 = Adafruit_VL53L0X();
Adafruit_VL53L0X r = Adafruit_VL53L0X();
Adafruit_VL53L0X r45 = Adafruit_VL53L0X();
Adafruit_VL53L0X fr = Adafruit_VL53L0X();
Adafruit_VL53L1X dn = Adafruit_VL53L1X();
*/
Adafruit_VL53L0X lox[6]={
  Adafruit_VL53L0X(),Adafruit_VL53L0X(),
  Adafruit_VL53L0X(),Adafruit_VL53L0X(),
  Adafruit_VL53L0X(),Adafruit_VL53L0X()
};
VL53L0X_RangingMeasurementData_t measure[6];
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
long distance[6];
int m_pin[5]={ 2 , 0 , 4 , 6 , 8 };// 4 -> 8; 3 -> 6; 2 -> 4; 1 -> 2; 0 -> 0;
void setSensorAddress(){
  for(int p=10; p<=15; p++){
    pwm.setPWM(p, 0, 4096);
    delay(10);
  }
  Serial.print(" Booting Up L0X Sensors !");
  for(int p=10; p<=14; p++){
    pwm.setPWM(p, 4096, 0);
    int i=p-10;
    if(!lox[i].begin(0x30+i)){
      Serial.print(" Sensor L0X : ");
      Serial.print(i);
      Serial.print(" Not Booted Up ");
      Serial.println();
      while(1)delay(100);
    }
    delay(10);
  }
  Serial.print(F(" Done Booting Up L0X Sensors !\n\n"));
  delay(1000);
}

void grabDistances(){
  for(int si=0; si<=4; si++){
    lox[si].rangingTest(&measure[si], false);
    if(measure[si].RangeStatus != 4 | measure[si].RangeMilliMeter > 0){
      distance[si]=measure[si].RangeMilliMeter;
    }
    else{
      distance[si]=-1;
    }
  }
};

void mapMotors(){
  long l_pwm; // long pwm for applying to arduino mapping function
  for(int i=0; i<4; i++){
    l_pwm=0;
    if (distance[i]>0 && distance[i]<=2000){
      l_pwm=map( distance[i] , 1 , 500 , 1000 , 255 );
    };
    uint16_t m_pwm= l_pwm;// convert mapped motor pwm to uint16_t motor pwm;
    pwm.writeMicroseconds(m_pin[i], m_pwm);
    delay(5);
  }

  /*Check if user is about to walk  into a sudden drop*/
  l_pwm=0;
  if (distance[4]>=2000){
      l_pwm=map( distance[4] , 1 , 500 , 255 , 1000 );
      uint16_t m_pwm= l_pwm;// convert mapped motor pwm to uint16_t motor pwm;

      /* Create Buzzing mattern in all motors when the height is too tall */
      for(int k=0 ; k<3 ; k++){
        for(int i=0 ; i<5 ; i++){
          pwm.writeMicroseconds(m_pin[i], m_pwm);
        }
        delay(100);
        for(int i=0 ; i<5 ; i++){
          pwm.writeMicroseconds(m_pin[i], 255);
        }
        delay(100);
      }
    };
};

void setup() {
  Serial.begin(115200);
  pwm.begin();
  Wire.setClock(400000);
  setSensorAddress();
}

void loop() {
  grabDistances();
  mapMotors();
  int dat=distance[4];
  Serial.println(dat);
  /*
  for(int si=0; si<4; si++){
    int dat=distance[si];
    Serial.print(dat);
    Serial.println();
  }
  */
  delay(10);
}