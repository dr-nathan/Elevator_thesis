import numpy as np
from matplotlib import pyplot as plt

zoning_lift1 = 6 * [None] + [1] + 2 * [2] + [6] + 5 * [2] + [14, 10, 17, 2, 8, 2, 14, 16, None]
zoning_lift2 = 6 * [None] + [1, 2, 11] + 3 * [2] + [6] + 3 * [2] + [6, 8, 2, 14, 8, 8, 16, 13]
zoning_lift3 = 8 * [None] + [2, 8, 2, 2, 8, 16, 16] + 3 * [2] + 2 * [6] + [2, 2, 16, 8]
zoning_lift4 = 7 * [None] + [5] + 4 * [2] + [10, 2, 6, 2, 14, 2, 10, 16, 16, 8] + 2 * [None]
zoning_lift5 = 7 * [None] + [1] + 4 * [2] + [12, 16, 2, 2, 14, 14, 2, 2, 16] + 3 * [None]
zoning_lift6 = 6 * [None] + [1, 12] + 3 * [2] + [8, 6, 2, 16, 14, 6, 6] + 4 * [16] + [None] + [13]

zones = np.asarray([zoning_lift1, zoning_lift2, zoning_lift3, zoning_lift4, zoning_lift5, zoning_lift6])

# separate plot for zoning matrix
fig2 = plt.figure()
# change figure size, for this plot only
fig2.set_figwidth(10)
# replace none with 17, change to int
zones = np.where(zones == None, 17, zones)
zones = zones.astype(int)
plt.imshow(zones, cmap='viridis')
# show number in each cell
for i in range(zones.shape[0]):
    for j in range(zones.shape[1]):
        floor = str(int(zones[i, j])) if zones[i, j] != 17 else 'X'
        # change color of cell if no zoning decision
        if floor == 'X':
            plt.text(j, i, floor, ha="center", va="center", color="red")
        else:
            plt.text(j, i, floor, ha="center", va="center", color="w")
# print ever 2 hours
plt.xticks(np.arange(0, 24, 2))
plt.title('Zoning matrix')
plt.ylabel('Elevator')
plt.xlabel('Hour')

plt.show()

# Richie, we are a concerned that we are running out of time. We understand that you need time to sort out
# your situation, but we are also in a position where we need to make decisions.
# We are a bit frustrated that the date is already pushed back from may to june, as we thought we
# had a deal for may that was confirmed. We are willing to wait until the end of may to give you some extra time,
# but we are not willing to wait any longer than that. this still gives you 1.5 months notice,
# and delaying the decision puts us in a really uncomfortable position as well.
# We would like to have your confirmation on sunday at the latest, so we can confirm with the tenants on monday.
#
