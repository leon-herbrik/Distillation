id = "something"
video = [0, 1, 2, 3, 4, 5, 6, 7]
context_size = 1
int_index = []
for i in range(0, len(video), context_size):
    if i + context_size > len(video):
        break
    int_index.append((id, i, i + context_size))

print(int_index)
