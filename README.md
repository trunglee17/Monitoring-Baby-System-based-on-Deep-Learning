Gmail: trung.le170920@gmail.com

Link Demo: 

# Monitoring-Baby-System-based-on-Deep-Learning

The monitoring of young children is one of the issues of interest of parents in a busy life, in this topic, our group will give ideas and implement solutions for monitoring young children while ensuring the movement of young children during sleep and wake up through the camera. 

In this project, the LSTM network will be built and combined with the Media Pipe library to automatically identify the child's facial features andCancel changes skeleton to detect behaviors such as:

**1. Detection baby wake up**

Calculate the value EAR based on 12 points marked on the child's eyes and train with LSTM network
![alt text](https://github.com/trunglee17/Monitoring-Baby-System-based-on-Deep-Learning/blob/main/image/eye.png)

**2. Detection baby moving**

The MediaPipe library will identify whether the child tends to be moving or not through the points on the skeleton marked by the library
![alt text](https://github.com/trunglee17/Monitoring-Baby-System-based-on-Deep-Learning/blob/main/image/body.png)
**3. Detection baby outside**

The system will let the user design and select a monitored region. The system will then use an algorithm to assess whether the child is initially in the area, ensuring that the child cannot leave the area under observation
![alt text](https://github.com/trunglee17/Monitoring-Baby-System-based-on-Deep-Learning/blob/main/image/outside.png)
