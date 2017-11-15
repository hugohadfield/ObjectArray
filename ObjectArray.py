
from functools import reduce
import operator
import copy

def prod(input_list):
    return reduce(operator.mul, input_list, 1)

class ObjectArray():

    # 
    # A pure python class for operating on dense arrays of objects that 
    # define their own multiplication and addition routines. 
    # 
    # 
    # Warning: very much a work in progress, likely very slow
    # 
    # 

    def __init__(self, nested_list):
        self.object_field = nested_list

    # Set up some useful properties
    @property
    def dimensionality(self):
        return self.__dimensionality

    @property
    def shape(self):
        return self.__shape

    @property
    def object_field(self):
        return self.__object_field

    @object_field.setter
    def object_field(self, nested_list):
        self.__object_field = copy.deepcopy(nested_list)
        self.__shape = ObjectArray.get_shape(nested_list)
        self.__dimensionality = len(self.__shape)

    @staticmethod
    def get_dimensionality(input_nested_list):
        current_list = input_nested_list[:]
        for dimension_index in range(20):
            current_list = current_list[dimension_index]
            if not isinstance(current_list, list):
                return dimension_index + 1

    @staticmethod
    def get_shape(input_nested_list):
        list_shape = []
        current_list = input_nested_list
        if not isinstance(current_list, list):
            return list_shape
        else:
            list_shape.append(len(current_list))
        for dimension_index in range(20):
            current_list = current_list[0]
            if not isinstance(current_list, list):
                return list_shape
            else:
                list_shape.append(len(current_list))

    # Set printing behaviour
    def __str__(self):
        return self.__object_field.__str__()

    def map(self, mapping_function):
        new_object = copy.deepcopy(self)
        n_elements = prod(self.shape)
        for iteration in range(n_elements):
            shaped_indices = get_convolution_indices(iteration,self.shape)
            value = get_matrix_value_by_indices(self.object_field,shaped_indices)
            mapped_value = mapping_function(value)
            set_matrix_value_by_indices(new_object.object_field, shaped_indices, mapped_value)
        return new_object

    # Override the infix operator for 'Matrix multiplication'
    def __matmul__(self, other):
        other_type = type(other)
        if (other_type is ObjectArray) or issubclass(other, ObjectArray):
            # Ensure we are operating on matching shaped matrices
            if (self.dimensionality == 2) and (other.dimensionality == 2):
                if (self.shape[1] == other.shape[0]):
                    # Do the actual matrix multiplication
                    a = self.object_field
                    b = other.object_field
                    # https://stackoverflow.com/questions/10508021/matrix-multiplication-in-python
                    zip_b = zip(*b)
                    zip_b = list(zip_b)
                    return ObjectArray( [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
                             for col_b in zip_b] for row_a in a] )
                else:
                    raise ValueError("Matrix shape mismatch")
            else:
                raise NotImplementedError("Only 2D matrix multiplication is implemented")
        elif other_type in [Integer, Float]:
            raise NotImplementedError("Mixed Integer/Float and matrix multiplication not yet implemented")
        else:
            raise TypeError("DON'T DO THIS")

    def __rmatmul__(self, other):
        # We know the other type is not an ObjectArray otherwise it would have had 
        # matmul called on it and we would not be here
        other_type = type(other)
        if other_type is Integer:
            raise NotImplementedError("Mixed Integer and matrix multiplication not yet implemented")
        else:
            raise NotImplementedError("Not sure what needs to be implemented here")

    def transpose(self):
        if self.dimensionality == 1:
            return ObjectArray([[val] for val in self.__object_field])
        elif self.dimensionality == 2:
            return ObjectArray([list(tup) for tup in zip(*self.__object_field)])
        else:
            raise NotImplementedError("High dimensional transposition is not implemented")

    def flip_in_axis(self,axis=0):
        return ObjectArray(flip_in_axis(self.object_field,axis))

    # We have to define two convolution operators as we have no guarantees of commutivity
    def left_convolve(self, convolution_kernel):
        # Does the convolution operation with left multiplies of the filter
        return self.__convolve(convolution_kernel,LR_flag=0)

    def right_convolve(self, convolution_kernel):
        # Does the convolution operation with right multiplies of the filter
        return self.__convolve(convolution_kernel,LR_flag=1)

    def __convolve(self,convolution_kernel,LR_flag=0):
        # Flip the convolution_kernel and run it as a correlation
        other_type = type(convolution_kernel)
        flipped_kernel = []
        if other_type is ObjectArray or issubclass(other_type, ObjectArray):
            flipped_kernel = copy.deepcopy(convolution_kernel)
            for i in range(convolution_kernel.dimensionality):
                flipped_kernel = flipped_kernel.flip_in_axis(i)
        elif other_type is List:
            for i in range(convolution_kernel.dimensionality):
                flipped_kernel = flip_in_axis(flipped_kernel,i)
            flipped_kernel = ObjectArray(convolution_kernel)
        else:
            raise InputError('Must be a list or ObjectArray')
        return self.__correlate(flipped_kernel,LR_flag)

    def __correlate(self,convolution_kernel,LR_flag=0):
        other_type = type(convolution_kernel)
        # We don't zero pad as we don't assume anything about the type of object we hold
        if other_type is ObjectArray:
            # Check the they have the same dimensionality...
            if self.dimensionality == convolution_kernel.dimensionality:

                # Get the shapes so we can process
                input_shape = self.shape
                kernel_shape = convolution_kernel.shape
                iteration_limits = [input_shape[i] - kernel_shape[i] + 1 for i in range(len(input_shape))]

                # Get the size of the convolution kernel
                n_kernel_elements = prod(kernel_shape)

                # Allocate output memory
                output_values = create_empty_nested_list(iteration_limits)

                for input_iteration in range(prod(iteration_limits)):
                    # Get the top left 
                    top_left_index_list = get_convolution_indices(input_iteration,iteration_limits)
                    accumulator = None
                    for kernel_iteration in range(n_kernel_elements):

                        # Get the kernel indices and actual indicies
                        kernel_indices = get_convolution_indices(kernel_iteration,kernel_shape)
                        input_index_list = [top_left_index_list[i] + kernel_indices[i] for i in range(len(kernel_indices))]
                        
                        # Get the kernel item and the input item
                        input_item = get_matrix_value_by_indices(self.__object_field, input_index_list)
                        kernel_item = get_matrix_value_by_indices(convolution_kernel.object_field, kernel_indices)

                        # Finally do the multiplication and addition
                        if LR_flag == 0:
                            # Left 
                            if accumulator is not None:
                                accumulator += (kernel_item*input_item)
                            else:
                                accumulator = (kernel_item*input_item)
                        else:
                            # Right 
                            if accumulator is not None:
                                accumulator += (input_item*kernel_item)
                            else:
                                accumulator = (input_item*kernel_item)

                    # Store our value
                    set_matrix_value_by_indices(output_values, top_left_index_list, accumulator)
                return ObjectArray(output_values)

        elif other_type is List:
            return self.__correlate(ObjectArray(convolution_kernel),LR_flag)
        else:
            raise TypeError("DON'T DO THIS")


def create_empty_nested_list(shape):
    repeated_object = [None for i in range(shape[-1])]
    for dimension_size in list(reversed(shape))[1:]:
        repeated_object = [copy.deepcopy(repeated_object) for i in range(dimension_size)]
    return repeated_object

def get_convolution_indices(iteration,shape):
    reversed_shape = list(reversed(shape))
    dimensionality = len(shape)
    if dimensionality == 1:
        return [iteration]
    else:
        indicies = shape[:]
        remaining_iterations = iteration
        for n in range(len(shape)):
            indicies[n] = remaining_iterations % reversed_shape[n]
            remaining_iterations = (remaining_iterations -indicies[n])//reversed_shape[n]
        return list(reversed(indicies))

def flip_in_axis(nested_list, axis=0):
    if axis == 0:
        return list(reversed(nested_list))
    else:
        return [flip_in_axis(nested_list_element, axis-1) for nested_list_element in nested_list]

def set_matrix_value_by_indices(nested_list, index_list, value):
    inner_list_reference = reduce(operator.getitem, index_list[:-1], nested_list)
    inner_list_reference[index_list[-1]] = value

def get_matrix_value_by_indices(nested_list, input_index_list):
    object_reference =  reduce(operator.getitem, input_index_list, nested_list)
    return object_reference



def run_tests():
    test_shape_and_dimensionality()
    test_infix_override()
    test_indexing()
    test_convolution()
    test_axis_flipping()
    test_map()

def test_shape_and_dimensionality():
    print(' ')
    print('############ test_shape_and_dimensionality ############')
    nested_array = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]]

    nd = ObjectArray.get_dimensionality( nested_array )
    print('Raw dimensionality check ',nd)

    shape = ObjectArray.get_shape( nested_array )
    print('Raw shape check ',shape)

    test_object = ObjectArray(nested_array)
    shape = test_object.shape
    dimensionality = test_object.dimensionality
    print('Instantiated dimensionality check ',nd)
    print('Instantiated shape check ',shape)

def test_infix_override():
    print(' ')
    print('############ test_infix_override ############')
    test_object_one = ObjectArray([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    test_object_two = ObjectArray([1,1,1]).transpose()
    print('input shape check ',test_object_one.shape, test_object_two.shape)

    output_obj = test_object_one@test_object_two
    print('Product dimensionality check ',output_obj.dimensionality)
    print('Product shape check ',output_obj.shape)
    print(output_obj)

def test_indexing():
    print(' ')
    print('############ test_indexing ############')
    empty_list = create_empty_nested_list([3,2,4])
    print(empty_list)
    set_matrix_value_by_indices(empty_list, [1,1,2],1)
    print(empty_list)
    print(  get_convolution_indices(7,[3,2,4]) )

def test_axis_flipping():
    print(' ')
    print('############ test_axis_flipping ############')
    input_mat = [ [[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]] ]
    print('Input matrix ',input_mat)
    axis = 2
    flipped_mat = flip_in_axis(input_mat,axis)
    print('Flipped matrix ',flipped_mat)

    input_obj = ObjectArray(input_mat)
    flipped_obj = input_obj.flip_in_axis(axis)
    print('Flipped matrix object',flipped_obj)

def test_convolution():
    print(' ')
    print('############ test_convolution ############')
    input_image = ObjectArray([[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2],[1,1,1,1,2,2,2,2]])
    convolution_kernel = ObjectArray([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])

    output_object = input_image.left_convolve(convolution_kernel)
    print(output_object)

def test_map():
    print(' ')
    print('############ test_map ############')
    input_obj = ObjectArray([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]])
    print(' Multiply ')
    print('Map input: ',input_obj)
    print('Map output: ',input_obj.map(lambda x: x*2))
    print('Map input still the same: ',input_obj)
    print(' Add ')
    print('Map input: ',input_obj)
    print('Map output: ',input_obj.map(lambda x: x + 2))
    print('Map input still the same: ',input_obj)

if __name__ == '__main__':
    run_tests()
