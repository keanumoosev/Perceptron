import numpy as np
import matplotlib.pyplot as plt
x=np.array([[2,3],
            [1,5],
            [2,8],
            [5,1],
            [6,2],
            [7,3]
            ])
y=np.array([0,0,0,1,1,1])
weights=np.random.rand(2)
bias=np.random.rand(1)
learning_rate=0.01
def step(x):
    return 1 if x>=0 else 0
for epoch in range(100):
    total_error=0
    for i in range(len(x)):
        linear_output=np.dot(x[i],weights)+bias
        prediction=step(linear_output)
        error=y[i]-prediction
        weights+=learning_rate*error*x[i]
        bias+=learning_rate*error
        total_error+=abs(error)
    if total_error==0:
        break
print("weights",weights)
print("bias",bias)
for i in range(len(x)):
    if y[i]==0:
        plt.scatter(x[i][0],x[i][1],color='red')
    else:
        plt.scatter(x[i][0],x[i][1],color='blue')
x_values=np.array([0,8])
y_values= -(weights[0]/weights[1])*x_values-(bias/weights[1])
plt.plot(x_values, y_values,color='green',label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('perceptron decision boundary')
plt.legend
plt.grid(True)
plt.show()
