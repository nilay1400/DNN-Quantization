import torch
import random
import math
import struct

class FI():
    def __init__(self, x):
        self.weight = x
        dense_tensor = self.weight.to_dense()
        self.x_size, self.z_size, self.y_size, _ = dense_tensor.shape
        unfolded_list = list(dense_tensor)
        unfolded_tensor = torch.cat(unfolded_list)
        self.flattened = unfolded_tensor.flatten()
        flatten_numpy = self.flattened.detach().numpy()
        param = flatten_numpy.size
        Sufficient_number_of_faults = round((31 * param) / (1 + ((31 * param) - 1) * (0.00010412328)))
        Sufficient_number_of_faults = math.floor(Sufficient_number_of_faults * 0.1)

    def param (self, x):
        self.weight = x
        dense_tensor = self.weight.to_dense()
        unfolded_list = list(dense_tensor)
        unfolded_tensor = torch.cat(unfolded_list)
        self.flattened = unfolded_tensor.flatten()
        flatten_numpy = self.flattened.detach().numpy()
        param = flatten_numpy.size
        return(param)


    def flip_random_bit(self, float_value, iibit):
        int_value = math.floor(float_value)

        # Flip the bit
        if iibit == 7:
            if int_value == -127:
                self.flipped_value = 0
            else:
                self.flipped_value = int_value * -1
        #elif iibit == 2 or iibit ==3:
            #self.flipped_value = int_value
        else:
            self.flipped_value = int_value ^ (1 << iibit)
        return self.flipped_value

    def fault_position(self):

        self.random_index = random.randint(0, self.flattened.shape[0] - 1)

        # Generate a random bit position to flip
        self.bit_position = random.randint(0, 4)

        return self.random_index, self.bit_position


    def inject(self, iindex, ibit):

        flattened_copy = self.flattened.clone()
            # print(flattened_copy)
        #print("random in flattened", iindex)
        #print(flattened_copy[iindex])
        faulty_weight_float = self.flip_random_bit(flattened_copy[iindex], ibit)
        faulty_weight = torch.tensor(faulty_weight_float)
        flattened_copy[iindex] = faulty_weight
        #print(flattened_copy[iindex])

        modified_weights = flattened_copy.reshape(self.x_size, self.z_size, self.y_size, self.y_size)
        self.updated_weights = modified_weights.to(self.weight.dtype)
        return self.updated_weights
