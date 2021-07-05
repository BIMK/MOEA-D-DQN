import numpy as np
import matplotlib.pyplot as plt
x = range(1, 13, 1)
y = range(1, 13, 1)
plt.plot(x, y)
plt.xticks(range(1, 13), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue', 'Lily', 'Ava', 'Isla', 'Rose', 'Jack', 'Leo', 'Charlie'))
# plt.xticks(x, ('Tom', 'Dick', 'Harry', 'Sally', 'Sue', 'Lily', 'Ava', 'Isla', 'Rose', 'Jack', 'Leo', 'Charlie'))
plt.show()
