# Convert the data to numpy arrays explicitly to ensure correct type handling
import numpy as np
import matplotlib.pyplot as plt

number_of_epochs = 19
epoch = np.array(list(range(number_of_epochs)))

train_mean_psnr_02_no_mask = np.array([26.54260026, 27.38554743, 27.74507388, 27.89018938, 28.03475244,
                                       28.15241487, 28.24323091, 28.25464221, 28.32789357, 28.369523,
                                       28.41615413, 28.45272024, 28.46642279, 28.53628659, 28.54102105,
                                       28.56251002, 28.6027613, 28.64482192], dtype=float)

train_std_psnr_02_no_mask = np.array([1.34781525, 1.25319197, 1.25371829, 1.26478982, 1.25020266,
                                      1.22957066, 1.23779882, 1.2496298, 1.23986232, 1.2431954,
                                      1.25985942, 1.25169165, 1.26606835, 1.24276305, 1.23479938,
                                      1.25632416, 1.25178423, 1.23670219], dtype=float)

test_mean_psnr_02_no_mask = np.array([24.75877482, 26.10052841, 26.55245602, 26.33894254, 25.29543094,
                                      26.88797103, 26.88406811, 26.93991903, 26.81778889, 26.9756608,
                                      27.08693369, 27.05071895, 27.14205259, 27.12687075, 27.0721751,
                                      27.0400802, 27.02131121, 27.23100068], dtype=float)

test_std_psnr_02_no_mask = np.array([1.65062217, 1.75796802, 1.66839898, 1.6202175, 1.58163607,
                                     1.65215359, 1.63973644, 1.6627144, 1.6791423, 1.65242042,
                                     1.67423327, 1.70945668, 1.66256567, 1.62554969, 1.63295569,
                                     1.64966688, 1.66489814, 1.6395492], dtype=float)

train_mean_psnr_02_learn_mask = np.array([29.75534022, 31.1071955, 31.48746649, 31.61441559, 31.81663926,
                                          31.97751342, 32.15362195, 32.20944608, 32.3869943, 32.45387534,
                                          32.52211739, 32.60985719], dtype=float)
train_std_psnr_02_learn_mask = np.array([1.53163745, 1.36925589, 1.38770808, 1.39279608, 1.42085827,
                                         1.36185548, 1.31748667, 1.36652947, 1.39233083, 1.4013921,
                                         1.36386729, 1.3927572], dtype=float)

test_mean_psnr_02_learn_mask = np.array([29.7021015, 29.61329425, 30.43234531, 30.61659358, 30.22504518,
                                         30.8455571, 30.77755851, 31.00253723, 31.20931219, 30.98162111,
                                         31.2572593, 31.32835521], dtype=float)
test_std_psnr_02_learn_mask = np.array([1.72461382, 1.67091067, 1.86569239, 1.89461506, 1.80780546,
                                        1.93895247, 1.93145603, 1.97754132, 2.01636847, 1.93259443,
                                        2.01681603, 2.03575813], dtype=float)


train_mean_psnr_04_learn_mask = np.array([
    27.28256088, 29.20395422, 29.94893574, 30.22488182, 30.48809678,
    30.6933827, 30.90490686, 30.99919843, 31.17499635, 31.28684521,
    31.39926501, 31.45789746, 31.50227924, 31.5917101, 31.66062225,
    31.68183099, 31.71718896, 31.81180301
], dtype=float)
train_std_psnr_04_learn_mask = np.array([
    1.75761577, 1.41101284, 1.38613093, 1.38421704, 1.37580998,
    1.38626777, 1.36044709, 1.41475292, 1.3987447, 1.40815844,
    1.43225083, 1.3796396, 1.40346738, 1.40090841, 1.38985275,
    1.42519957, 1.46535286, 1.36219635
], dtype=float)
test_mean_psnr_04_learn_mask = np.array([
    26.68073885, 28.45699092, 29.01910856, 29.42969699, 29.02506803,
    29.54488199, 25.64196922, 29.81363258, 29.97138166, 29.46111494,
    30.15297914, 30.29538165, 30.40930838, 30.44407258, 30.44206692,
    30.37872189, 30.52589701, 30.64634208
], dtype=float)
test_std_psnr_04_learn_mask = np.array([
    1.5025399, 1.62261517, 1.69027567, 1.76219765, 1.68808312,
    1.75498935, 1.78240147, 1.79727452, 1.82941392, 1.69154021,
    1.85274453, 1.88448378, 1.90905502, 1.92418447, 1.91043263,
    1.90036779, 1.92896811, 1.95406821
], dtype=float)

train_mean_psnr_04_no_mask = np.array([23.95583887, 25.17007845, 25.538796, 25.69611637, 25.84777588,
                                       25.99330665, 26.08110698, 26.10276502, 26.18939541, 26.223331,
                                       26.27684525, 26.31742259, 26.33758322, 26.4079114, 26.41764608,
                                       26.43723211, 26.480241085], dtype=float)
train_std_psnr_04_no_mask = np.array([1.38012978, 1.23921186, 1.23189319, 1.24686023, 1.24292225,
                                      1.20746353, 1.21771789, 1.23238812, 1.23077545, 1.24023135,
                                      1.22823838, 1.24247836, 1.24788509, 1.23146472, 1.22452173,
                                      1.23651094, 1.24633195], dtype=float)
test_mean_psnr_04_no_mask = np.array([23.49150033, 24.34569286, 22.54304188, 24.53480456, 22.94507667,
                                      24.35635036, 24.65712693, 24.89162413, 24.63475011, 24.85944191,
                                      24.89483383, 24.7707914, 24.99212634, 24.98597695, 24.8440928,
                                      25.04587172, 25.00568839], dtype=float)
test_std_psnr_04_no_mask = np.array([1.73120538, 1.63037304, 1.58257291, 1.61189887, 1.59917586,
                                     1.74268075, 1.60967492, 1.66804129, 1.62596865, 1.64195166,
                                     1.60534922, 1.73121811, 1.60246862, 1.59544516, 1.53270291,
                                     1.62185978, 1.573526502], dtype=float)

train_mean_psnr_06_learn_mask = np.array([25.82321186, 27.92042751, 28.77055125, 29.19659777, 29.52165878,
                                         29.80165926, 30.00322545, 30.10612173, 30.23919700, 30.36829728,
                                         30.48682943, 30.56163159, 30.66749096], dtype=float)
train_std_psnr_06_learn_mask = np.array([1.83741400, 1.37775006, 1.32090722, 1.35138743, 1.38488201,
                                        1.37107413, 1.35507249, 1.35147883, 1.38549159, 1.38066102,
                                        1.36324939, 1.37435713, 1.38163821], dtype=float)
test_mean_psnr_06_learn_mask = np.array([25.90950875, 27.00623997, 27.92828265, 25.43095257, 27.76947723,
                                         28.73140275, 28.89122024, 28.98116400, 29.11909445, 28.90957383,
                                         29.40291692, 29.21993769, 29.35582298], dtype=float)
test_std_psnr_06_learn_mask = np.array([1.42355441, 1.53055586, 1.58361730, 1.78386402, 1.54725396,
                                        1.65402609, 1.71320358, 1.71330625, 1.72011676, 1.66728697,
                                        1.78211708, 1.74963130, 1.76597838], dtype=float)

train_mean_psnr_06_no_mask = np.array([21.57201513, 22.72416686, 23.1088933, 23.25840007, 23.39085589,
                                       23.52371234, 23.60182279, 23.63161543, 23.71173239, 23.75370909,
                                       23.81125299, 23.85541113, 23.87869865, 23.95006166, 23.95831155,
                                       23.98410068, 24.0197426, 24.05122752, 24.07635672], dtype=float)
train_std_psnr_06_no_mask = np.array([1.3287265, 1.23385864, 1.22653383, 1.25299015, 1.23543173,
                                      1.21688579, 1.21425098, 1.24654522, 1.25056735, 1.24168799,
                                      1.2329958, 1.24430737, 1.25244131, 1.24056423, 1.23428011,
                                      1.24231723, 1.24595543, 1.23130026, 1.21372955], dtype=float)
test_mean_psnr_06_no_mask = np.array([21.20571274, 20.86479229, 22.01358681, 21.6346603, 22.09753873,
                                      22.06804963, 22.2878964, 22.30573702, 22.1346829, 22.20782719,
                                      22.44682902, 22.24181853, 22.32805404, 22.40480298, 22.44807659,
                                      22.3277418, 22.03850544, 22.4752513, 22.55606352], dtype=float)
test_std_psnr_06_no_mask = np.array([1.39056104, 1.37969061, 1.49364439, 1.63158495, 1.55575304,
                                     1.49952493, 1.54446374, 1.54202918, 1.48899579, 1.57525044,
                                     1.54365397, 1.49770894, 1.51565632, 1.51323021, 1.55912914,
                                     1.54126605, 1.50389836, 1.54575503, 1.48975606], dtype=float)

train_mean_psnr = train_mean_psnr_06_no_mask
train_std_psnr = train_std_psnr_06_no_mask
test_mean_psnr = test_mean_psnr_06_no_mask
test_std_psnr = test_std_psnr_06_no_mask

# Plotting
plt.figure(figsize=(12, 8))

# Colors
train_color = '#1f77b4'
test_color = '#ff7f0e'

# option 1
plt.plot(epoch, train_mean_psnr, label='Train Mean PSNR',
         marker='o', linestyle='-')
plt.plot(epoch, test_mean_psnr, label='Validation Mean PSNR',
         marker='o', linestyle='-')

# Std PSNR as shaded areas
plt.fill_between(epoch, train_mean_psnr - train_std_psnr,
                 train_mean_psnr + train_std_psnr, alpha=0.15)
plt.fill_between(epoch, test_mean_psnr - test_std_psnr,
                 test_mean_psnr + test_std_psnr, alpha=0.15)

# option 2
# plt.errorbar(epoch, train_mean_psnr, yerr=train_std_psnr, label='Train Mean PSNR',
#              marker='o', linestyle='-', capsize=5, capthick=2, color=train_color)
# plt.errorbar(epoch, test_mean_psnr, yerr=test_std_psnr, label='Test Mean PSNR',
#              marker='o', linestyle='-', capsize=5, capthick=2, color=test_color)

# option 3:
# Mean PSNR
# plt.plot(epoch, train_mean_psnr, label='Train Mean PSNR',
#          marker='o', linestyle='-', color=train_color)
# plt.plot(epoch, test_mean_psnr, label='Test Mean PSNR',
#          marker='o', linestyle='-', color=test_color)

# Dashed lines for std bounds
plt.plot(epoch, train_mean_psnr + train_std_psnr,
         linestyle='--', color=train_color, alpha=0.5)
plt.plot(epoch, train_mean_psnr - train_std_psnr,
         linestyle='--', color=train_color, alpha=0.5)
plt.plot(epoch, test_mean_psnr + test_std_psnr,
         linestyle='--', color=test_color, alpha=0.5)
plt.plot(epoch, test_mean_psnr - test_std_psnr,
         linestyle='--', color=test_color, alpha=0.5)


# Adding titles and labels
title2 = "Drop rate = 0.6 without using learnable mask"
plt.title("Mean and Std PSNR per Epoch: " + title2, fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.legend()
plt.grid(True)
plt.show()
