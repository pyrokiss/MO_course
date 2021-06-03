import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate

alpha = [2,4,6,8,10]
N = [2,6,10,12,16]
M = 1
n = 1
epsilon = 0.05
"""
**Функція 1**. Дана функція відповідає за задану функцію в завданні.
"""

def function(x):
  func = 100*(((x[0]**2)-x[1])**2)+(x[0]-1)**2 #2*((x1 - 3)**2)+(x2-4)**2
  return func

"""**Функція 2**. Ця функція відповідає за побудови графіків. Функція реалізує графік для кожної ітерації програми. За замовчуванням функція будує 3 графіка, але для оптимізації, ми виводимо лише останній графік."""

def drow_function(x1, x2, x3, new = None):
  #plt.figure(figsize=(20, 8))
  plt.plot([x1[0],x2[0]], [x1[1],x2[1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)
  plt.plot([x1[0],x3[0]], [x1[1],x3[1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)
  plt.plot([x2[0],x3[0]], [x2[1],x3[1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)
  if new is not None:
    plt.plot([x1[0],new[0]], [x1[1], new[1]], "-", c="grey", marker = "o", markerfacecolor='#1e3d59', markersize=13)
    plt.plot([x2[0],new[0]], [x2[1], new[1]], "-", c="grey", marker = "o", markerfacecolor='#1e3d59', markersize=13)
    plt.plot([x3[0],new[0]], [x3[1], new[1]], "--", c="grey", marker = "o", markerfacecolor='#1e3d59', markersize=13)
    plt.scatter(new[0], new[1], c="blue")
  plt.grid(True)
  #plt.show()

"""**Функція 3**. Ця функція необхідна для сортування $x^0, x^1, x^2$."""

def sort_function(x, f):
  indices = f.argsort()
  x_a = x[indices]
  f_a = f[indices]

  min = x_a[0]
  middle = x_a[1]
  max = x_a[2]

  return min,middle,max

"""**Функція 4**. Це головна функція, яка відповідає за метод пошуку за симплексом. Це рекурсивна функція, тобто для кожної ітерації, функція викликає саму себе. В функції є перевірки стосовно $\varepsilon$ та різниця між мінімальним та середнім значеннями функцій."""

def simplex_function(min, middle, max, n, M, alpha, N):
  x_array.append([list(min), list(middle), list(max)])
  if function(min) > epsilon and (function(middle)-function(min))>0.00001:#if n <= 15:
    if M <= 3:
      print("Итерация №{}:".format(n))
      print("\tmin: {} => f(min): {}".format(min, function(min)))
      print("\tmiddle: {} => f(middle): {}".format(middle, function(middle)))
      print("\tmax: {} => f(max): {}".format(max, function(max)))
      plt.figure(figsize=(20, 8))
      plt.subplot(1, 3, 1)
      drow_function(min, middle, max)
      mid = (min+middle)/2
      print("\tx_mid: {}".format(mid))
      plt.subplot(1, 3, 2)
      drow_function(min, middle, max, mid)


      x_new = 2*mid-max
      print("\tx_new: {}".format(x_new))
      plt.subplot(1, 3, 3)
      drow_function(min, middle, max, x_new)
      plt.show()
      min_, middle_, max_ = sort_function(np.array([min,middle,x_new]), np.array([function(min),function(middle),function(x_new)]))
      if middle_.all() == middle.all():
        M+=1
        n+=1
        simplex_function(min_, middle_, max_, n, M, alpha, N)
      else:
        n+=1
        simplex_function(min_, middle_, max_, n, M, alpha, N)
    else:
      print("===Произведем уменьшение alpha в 2 раза===")
      
      alpha = alpha/2
      x_0 = min
      delta_1 = ((np.sqrt(N + 1) + N - 1)/(N * np.sqrt(2)))*alpha
      delta_2 = ((np.sqrt(N + 1) - 1)/(N * np.sqrt(2)))*alpha
      print("delta_1: ", delta_1)
      print("delta_2: ", delta_2)
      x_1 = [x_0[0]+delta_2, x_0[1]+delta_1]#[7.0, 6.0]
      x_2 = [x_0[0]+delta_1, x_0[1]+delta_2]
      M = 1
      x_arr = np.array([x_0, x_1, x_2])
      f_arr = np.array([function(x_0), function(x_1), function(x_2)])
      min,middle,max = sort_function(x_arr, f_arr)
      simplex_function(min, middle, max, n, M, alpha, N)
  else:
    print("Количество итераций: {}".format(n))
    print("min: {} => f(min): {}".format(min, function(min)))
    temp.append(function(min))
    temp.append(n)

"""**Функція 5**. Ця функція необхідна для побудови остаточного графіку. Реалізовано це за допомогою списка, який заповнюєть координатами після кожної ітерації, і в кінці виводить самий графік.

"""

def all_graph_function(arr):
  plt.figure(figsize=(20, 8))
  for i in arr:
    plt.plot([i[0][0],i[1][0]], [i[0][1],i[1][1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)
    plt.plot([i[0][0],i[2][0]], [i[0][1],i[2][1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)
    plt.plot([i[1][0],i[2][0]], [i[1][1],i[2][1]], c="#ff6e40", marker = "o", markerfacecolor='#1e3d59', markersize=13)

  plt.grid(True)
  plt.show()

table = []
header = ["alpha", "N", "x0", "x1", "x2", "f(min)","n of iterations", "time", ]
x_0 = [-1.2, 0.0]
for i in range(len(alpha)):
  print("---alpha: {} ---N: {} ---".format(alpha[i], N[i]))
  start_time = time.time()
  delta_1 = ((np.sqrt(N[i] + 1) + N[i] - 1)/(N[i] * np.sqrt(2)))*alpha[i]
  delta_2 = ((np.sqrt(N[i] + 1) - 1)/(N[i] * np.sqrt(2)))*alpha[i]

  print(delta_1)
  print(delta_2)

  x_1 = [x_0[0]+delta_2, x_0[1]+delta_1]#[7.0, 6.0]
  x_2 = [x_0[0]+delta_1, x_0[1]+delta_2]

  x_arr = np.array([x_0, x_1, x_2])
  f_arr = np.array([function(x_0), function(x_1), function(x_2)])

  x_array = []
  for g, j in enumerate(x_arr):
    print("x{}: {} => f({}): {}".format(g, j, j, f_arr[g]))

  min,middle,max = sort_function(x_arr, f_arr)

  temp = [alpha[i], N[i], x_0, x_1, x_2]
  simplex_function(min, middle, max, n, M, alpha[i], N[i])
  #print(x_array)
  

  all_graph_function(x_array)
  end_time = time.time()
  temp.append((end_time - start_time))
  #print(temp)
  print("Время на выполнение: {} секунд.". format(end_time - start_time))
  table.append(temp)

print(tabulate(table, headers=header))

