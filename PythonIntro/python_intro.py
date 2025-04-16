# python_intro.py
"""Python Essentials: Introduction to Python.
<Name> Benjamin McElroy
<Class> MTH 420 Modern Methods Applied Math 
<Date> Friday, April 11th. 
"""


# Problem 1 (write code below)
if __name__ == "__main__":
    print("Hello, world!") # Indent with four spaces (NOT a tab).

# Problem 2
def sphere_volume(r):
    """ Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    pi = 3.14159
    volume = (4 / 3) * pi * (r ** 3)
    return volume

if __name__ == "__main__":
   
    test_radius = 5
    result = sphere_volume(test_radius)
    print(f"The volume of a sphere with radius {test_radius} is {result}")
    
    


# Problem 3
def isolate(a, b, c, d, e):
    """ Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, b, c, sep='     ', end=' ')
    print(d, e)
if __name__ == "__main__":
    isolate("a", "b", "c", "d", "f")

    
    


# Problem 4
def first_half(my_string):
    """ Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    return my_string[:len(my_string) // 2]
    
    

def backward(my_string):
    """ Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]
if __name__ == "__main__":
    text = "Python"
    print("First half:", first_half(text))   
    print("Backward:", backward(text))       
    
    
    
    


# Problem 5
def list_ops():
    """ Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    animals = ["bear", "ant", "cat", "dog"]
    animals.append("eagle")                      
    animals[2] = "fox"                           
    animals.pop(1)                               
    animals.sort(reverse=True)                   
    eagle_index = animals.index("eagle")
    animals[eagle_index] = "hawk"                
    animals[-1] += "hunter"                      
    return animals
if __name__ == "__main__":
    result = list_ops()
    print(result)

    


# Problem 6
def pig_latin(word):
    """ Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = "aeiouAEIOU"
    if word[0] in vowels:
        return word + "hay"
    else:
        return word[1:] + word[0] + "ay"
if __name__ == "__main__":
    print(pig_latin("apple"))  
    print(pig_latin("banana")) 

    

# Problem 7
def palindrome():
    """ Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """
    max_palindrome = 0
    for i in range(999, 99, -1):
        for j in range(i, 99, -1):  
            product = i * j
            if str(product) == str(product)[::-1] and product > max_palindrome:
                max_palindrome = product
    return max_palindrome
if __name__ == "__main__":
    result = palindrome()
    print("The largest palindrome made from the product of two 3-digit numbers is:", result)

    

# Problem 8
def alt_harmonic(n):
    """ Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    return sum([(-1) ** (k + 1) / k for k in range(1, n + 1)])
if __name__ == "__main__":
    approx_ln2 = alt_harmonic(500_000)
    print(f"Approximation of ln(2) with 500,000 terms: {approx_ln2}")

   