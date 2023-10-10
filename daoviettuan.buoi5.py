import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import pandas as pd
import tkinter as tk
from numpy import array
tf.disable_v2_behavior()

def predict_result():
    prediction_label.configure(text="Dự đoán: ...")
window = tk.Tk()
window.title("Dự đoán kết quả học tập")
input_label = tk.Label(window, text="Nhập dữ liệu:")
input_label.pack()

input_entry = tk.Entry(window)
input_entry.pack()

predict_button = tk.Button(window, text="Dự đoán", command=predict_result)
predict_button.pack()

prediction_label = tk.Label(window, text="Dự đoán: ")
prediction_label.pack()
window.mainloop()


# Đọc dữ liệu từ file CSV
df = pd.read_csv('Student_Performance.csv', index_col=0, header=0)
x = array(df.iloc[:900, :5])
y = array(df.iloc[:900, 5:6])

# Khởi tạo tập giá trị x và y
# x = np.linspace(0, 50, 50)
# y = np.linspace(0, 50, 50)
 
# Cộng thêm nhiễu cho tập x và y để có tập dữ liệu ngẫu nhiên
# x += np.random.uniform(-4, 4, 50)
# y += np.random.uniform(-4, 4, 50)

n = len(x)  # Số lượng dữ liệu

# Hiển thị dữ liệu huấn luyện
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

# Tạo model cho tập dữ liệu
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Khởi tạo biến w và b
W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

# Thiết lập tốc độ học
learning_rate = 0.01

# Số vòng lặp
training_epochs = 100

# Hàm tuyến tính
y_pred = tf.add(tf.multiply(X, W), b)

# Hàm mất mát Mean Squared Error
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

# Tối ưu bằng Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Thiết lập các biến toàn cục
init = tf.global_variables_initializer()

# Bắt đầu phiên làm việc với TensorFlow
with tf.Session() as sess:

    # Khởi tạo các biến
    sess.run(init)

    # Lặp qua từng epoch
    for epoch in range(training_epochs):

        # Đưa từng điểm dữ liệu vào optimizer bằng cách sử dụng Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

        # Hiển thị kết quả sau mỗi 50 epoch
        if (epoch + 1) % 50 == 0:
            # Tính toán giá trị hàm mất mát sau mỗi epoch
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))

    # Lưu các giá trị cần thiết để sử dụng bên ngoài phiên làm việc
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

# Tính toán dự đoán
predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =", weight, "Bias =", bias, '\n')

# Hiển thị kết quả
plt.plot(x, y, 'ro', label='Dữ liệu gốc')
plt.plot(x, predictions, label='Đường thẳng tìm được')
plt.title('Kết quả Hồi quy tuyến tính')
plt.legend()
plt.show()



