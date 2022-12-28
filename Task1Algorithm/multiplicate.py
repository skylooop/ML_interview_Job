import typing as tp
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_list("input_array", default=['1', '2', '3'], help="Input array.")

def multiply(_) -> tp.List[int]:
    '''
    This function runs in O(n) in time complexity. And in O(1) space complexity
    '''
    
    A: tp.List[int] = list(map(int, FLAGS.input_array))
    ans: tp.List[int] = [1] * len(A)
    
    running_left: int = 1
    running_right: int = 1
    
    for i in range(len(A)):
        ans[i] *= running_left
        running_left *= A[i]
        
        ans[-(i + 1)] *= running_right
        running_right *= A[-(i + 1)]

    return ans


if __name__ == "__main__":
    app.run(multiply)