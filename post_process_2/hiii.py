import matplotlib.pyplot as plt


def hii(unpaired_right_left, unpaired_up_down):

    full = unpaired_right_left + unpaired_up_down

    for [p1_x, p1_y, p2_x, p2_y], neighbor in full:
        dx = p2_x - p1_x
        dy = p2_y - p1_y
        plt.arrow(p1_x, p1_y, dx, dy, head_width=0.4,
                  head_length=0.7,
                  length_includes_head=True,
                  color='black')

    plt.gca().set_aspect('equal')
    plt.show()

