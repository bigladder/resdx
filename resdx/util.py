def calc_biquad(coeff, in_1, in_2):
  return coeff[0] + coeff[1]*in_1 + coeff[2]*in_1*in_1 + coeff[3]*in_2 + coeff[4]*in_2*in_2 + coeff[5]*in_1*in_2

def calc_quad(coeff, in_1):
  return coeff[0] + coeff[1]*in_1 + coeff[2]*in_1*in_1

def find_nearest(array, value):
  closest_diff = abs(array[0] - value)
  closest_value = array[0]
  for option in array:
    diff = abs(option - value)
    if diff < closest_diff:
      closest_diff = diff
      closest_value = option
  return closest_value
