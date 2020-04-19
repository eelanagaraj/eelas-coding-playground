### fun coding challenges wee!!
################################################################################
################################## HELPERS #####################################
################################################################################
def lst_pretty_print(l, tab_len=1, beg="[", end="]", sep="", line_num=False):
    """ just print out a list of lists, useful for matrices"""
    tab = "    "
    print("\n", beg)
    for i, lst in enumerate(l):
        if line_num:
            print(i),
        print(tab, lst, sep)
    print(end)
    return


def num_divisors_naive_deb(n, ret_list=False):
    """
    Returns the number of divisors of n
    When ret_list set to True, returns list of divisors, useful for debugging
    """
    # naive to check
    count = 0
    if ret_list:
        lst = []
    for i in range(n, 0, -1):
        if n % i == 0:
            count += 1
            if ret_list:
                lst.append(i)

    if ret_list:
        return count, lst
    return count


def interval_prime_sieve(block_start_i, block_sz):
    """ create a block where index 0 represents int block_start_i and the values
        in the block are True iff the corresponding index is prime """
    block = [True] * block_sz
    for i in range(2, block_start_i):
        # compute the offset from the first block
        offset = block_start_i % i
        if offset:
            offset = i - offset
        # sieve on the block
        for block_index in range(offset, block_sz, i):
            block[block_index] = False
    return block


def is_prime_sieve(n):
    """
    create a 0-indexed list where sieve[i] is true if i is prime, through n
    (size of list returned is n + 1)
    """
    # True if prime
    is_prime = [True] * (n + 1)
    # 1 is definitionally False

    is_prime[0:2] = [False, False]
    for i in range(2, n):
        for add in range(2 * i, n + 1, i):
            # mark multiples of i as not prime, excluding i itself
            is_prime[add] = False
    return is_prime


def prime_factorization(n, prime_lst, exp_only=False):
    """
    print n's prime factorization in a tuple list: (prime_num, exp)
    """
    # prime_lst = [ind for ind, is_p in enumerate(prime_sieve) if is_p]

    f_lst = []
    rem = n
    # iterate over prime lst and divide out if possible
    for prime in prime_lst:
        # if remaining value is 1 then done
        if rem == 1:
            break

        exp = 0
        while not (rem % prime):
            rem /= prime
            exp += 1
        if exp:
            if exp_only:
                f_list.append(exp)
            else:
                f_lst.append((prime, exp))

    return f_lst


def num_divs_pf(n, prime_lst):
    divs = 1
    rem = n
    # iterate over prime lst and divide out if possible
    for prime in prime_lst:
        # if remaining value is 1 then done
        if rem == 1:
            break

        exp = 0
        while not (rem % prime):
            rem /= prime
            exp += 1
        if exp:
            divs *= exp + 1
    return divs


def num_divisors(n):
    # only test to n/2 or n/3 + 1
    ceiling = n / 3 if n % 2 else n / 2

    count = 1
    for i in range(1, ceiling + 1):
        if n % i == 0:
            count += 1
    return count


def num_divisors_block(n):
    return [num_divisors(i + 1) for i in range(n)]


def num_divisors_sieve(n):
    sieve = [0] * n
    for x in range(n):
        for i in range(x, n, x + 1):
            sieve[i] += 1
    return sieve


def divisors_tab(n, proper=False):
    """ create divisors table up to n, sieve method
        performance: 4x faster than naive method below"""

    tab = [0 for i in range(n + 1)]
    for i in range(1, n + 1):
        start = i
        if proper:
            start *= 2
        for add in range(start, n + 1, i):
            tab[add] += i
    return tab


def factorial_p2_deprecated(x):
    """ reduce in python 2 deprecated in python 3"""
    # convention
    if x == 0:
        return 1
    return reduce(lambda a, b: a * b, range(1, x + 1))


def factorial(x):
    # convention
    if x == 0:
        return 1
    mult = 1
    for i in range(1, x + 1):
        mult *= i
    return mult


################################################################################
################################### SOLVED #####################################
################################################################################
def problem_1(limit=1000):
    threes_sum = sum(range(3, limit + 1, 3))
    fives_sum = sum(range(5, limit + 1, 5))
    double_count = sum(range(15, limit + 1, 15))

    return threes_sum + fives_sum - double_count


def problem_2_fib(limit=4000000):
    # compute fib terms as we go and also keep running sum, so one pass
    f_1 = 0
    f_2 = 1

    even_sum = 0
    while f_1 < limit:
        # only ever consider new computed values
        if not (f_1 % 2):
            even_sum += f_1
        # shift previous computed term down, use swap syntax
        f_1, f_2 = f_2, f_2 + f_1
        # compute new term
    return even_sum


def problem_2_try2(limit=4000000):
    """ attempted optimize for not checking even,
        but actually marginally slower? """
    f_1 = 0
    f_2 = 1

    even_sum = 0
    while f_1 < limit:
        # at start of iter, f_1 should always be even
        even_sum += f_1

        for _ in range(3):
            f_1, f_2 = f_2, f_2 + f_1
    return even_sum


def problem_3(num):
    """ Return list of prime factors of a number """
    prime_tab = {}

    def is_prime(n):
        # build up prime number table as we go to prevent repeat computation
        if not n in prime_tab:
            # build up table
            for x in range(2, n / 2 + 1):
                if not (n % x):
                    prime_tab[n] = False
            prime_tab[n] = True
        return prime_tab[n]

    factors = []
    factor = 2

    while factor <= num:
        # if it is a factor, check if it is also prime
        if not (num % factor):
            if is_prime(factor):
                factors.append(factor)
            num = num / factor
            factor = 2
        else:
            factor += 1
    return max(factors)


def is_palindrome(number):
    ## faster
    str_num = str(number)
    for i in range(len(str_num) // 2):
        if str_num[i] != str_num[-i - 1]:
            return False
    return True


def is_palindrome_div(number):
    # slower
    num_digits = int(math.log10(number)) + 1
    for i in range(num_digits / 2):
        dig_1 = number // 10 ** i % 10
        dig_2 = number // 10 ** (num_digits - 1 - i) % 10
        if dig_1 != dig_2:
            return False
    return True


def problem_4(limit=1000000):
    """ Find largest palindrome that is the product of 2 3-digit numbers"""
    # keeping track of tried vals is actually slower...
    # desc order = way less palindrome checks will be made
    # small prod of 3 palindrome
    max_val = 101 * 101
    for x in range(999, 101, -1):
        start_val = 999
        prod = x * start_val
        for _ in range(898):
            # see if prod could be the new max
            if prod > max_val:
                # check if palindrome
                if is_palindrome(prod):
                    max_val = prod

            # instead of mult, just add one more copy of x bc that's what multiplication is
            prod -= x
    return max_val


def problem_5(upper_bound=20):
    """ 2520 is the smallest number that can be divided by each of the numbers 
        from 1 to 10 without any remainder. What is the smallest positive number 
        that is evenly divisible by all of the numbers from 1 to 20?

    #### Math solve: just multiply factors shared by all digits 1-20: 232792560

    """

    # brute force/naive --> for each number starting with 2520, incr by 20

    found = False

    incr = upper_bound * (upper_bound - 1)
    nums_to_check = [x for x in range(1, upper_bound) if incr % x]
    print(nums_to_check)
    magic_int = 0
    while not found:
        magic_int += incr

        # check if all evenly divide (1,2,3,4,5,9,10 all covered by += 90)
        for x in range(6, 9):
            if magic_int % x == 0:
                # if last check
                if x == 8:
                    found = True
            else:
                break

    return magic_int


def problem_76(num=100):
    """
    Count the number of distinct ways (order doesn't matter) to sum 100
    Recurrence relation approach (using partition recurrence relation)
    ###### answer 190569291 ########
    """
    # for sake of organization
    def sum_of_divisors(n):
        """ e.g. for s_o_d(12) = 28 = 12 + 6 + 4 + 3 + 2  + 1"""
        t_sum = n
        for i in range(1, n):
            if n % i == 0:
                t_sum += i
        return t_sum

    def divisors_tab_naive(n):
        """ naive version for testing"""
        return [sum_of_divisors(i) for i in range(1, n + 1)]

    def p_inner(k, n, div_tab, p_tab):
        div = div_tab[n - k]  # check index
        p = p_tab[k]
        return div * p  # / float(n))

    def p(n=100):
        div_tab = divisors_tab(n)
        # set up p_tab, convention p(0) = 1
        p_tab = [0 for _ in range(n + 1)]
        p_tab[0] = 1

        for i in range(1, n + 1):
            for k in range(i):
                p_tab[i] += p_inner(k, i, div_tab, p_tab)
            p_tab[i] /= i
        return p_tab[n] - 1

    # compute solution
    return p(num)


def dp_sums(n, half_mat=True):
    """
    Dynamic Programming approach to solving the sums problem
    Note: could also do this using numpy, but it is already super fast so meh.

    For the n x n matrix, entry [i,j] represents the number of ways to sum to
    value i with lead (maximum) digit j. The diagonal is always 1, since there
    is only 1 way to sum to i using digit i, namely "i + 0 = i". Summing all
    the values in row i is the total unique ways to sum i including "i + 0 = i".

    To compute entry [i,j], j is the lead digit, and therefore the largest. We
    want to find the corresponding entry in the matrix which stores the number
    of ways to make the remaining portion of the sum (i - j) with maximum digit
    j, which is the sum of all entries in row i - j, col 1...j. Since the matrix
    is 0 indexed but the "values" represented are 1 indexed (i.e. natural nums),
    we need to subtract 1 (to ensure we are looking at the previous rows), and
    then keep in mind that we are including column j ([:j] excludes col j).

    Final answer just needs to subtract the 1 since this problem excludes "i + 0"
    as a valid sum. This function will just return the whole matrix

    """
    # would be lower triangular matrix, so we can omit the 0s in the top half
    sums_mat = [[0 for _ in range(i + 1)] for i in range(n)]
    # init diagonal to 1
    for i in range(n):
        sums_mat[i][i] = 1

    for i in range(n):
        # ignore the zeros after the diagonal
        for j in range(i):
            sums_mat[i][j] = sum(sums_mat[i - j - 1][: j + 1])

    # if you wanna see the final sums matrix
    lst_pretty_print(sums_mat, line_num=True)

    return sums_mat


def problem_76_dp(num=100):
    """ dynamic programming approach"""
    # solution, just sum last row of matrix, account for the "i + 0" case
    return sum(dp_sums(num)[num - 1]) - 1


################################################################################
################################# UNSOLVED #####################################
################################################################################


def problem_78(num=1000000, lim=2000):
    """similar to 76 but find first value that is divisible by num (1 mil)"""

    # may need to optimize using numpy

    n = lim
    block_sz = 100
    sums_mat = [[0 for _ in range(i + 1)] for i in range(n)]
    # init diagonal to 1
    for i in range(n):
        sums_mat[i][i] = 1
    i = 0
    while i < n:
        # ignore the zeros after the diagonal
        for j in range(i):
            sums_mat[i][j] = sum(sums_mat[i - j - 1][: j + 1])

        # check if divisible by num, if so return
        if sum(sums_mat[i]) % num == 0:
            return i + 1
        i += 1
        if i == n:

            print("adding block")
            a = [
                [1 if x == y - 1 else 0 for x in range(y)]
                for y in range(n + 1, n + block_sz + 1)
            ]
            sums_mat.extend(a)
            n += block_sz
        if i % 100 == 0:
            print(i)


def problem_6(limit=100):
    """The sum of the squares of the first ten natural numbers is,
    12 + 22 + ... + 102 = 385
    The square of the sum of the first ten natural numbers is,
    (1 + 2 + ... + 10)2 = 552 = 3025

    Hence the difference between the sum of the squares of the first ten natural
    numbers and the square of the sum is 3025 - 385 = 2640
    Find the difference between the sum of the squares of the first one hundred
    natural numbers and the square of the sum."""
    import numpy

    return sum(range(1, limit + 1)) ** 2 - sum(numpy.array(range(1, limit + 1)) ** 2)


def problem_7(n=10001, start_sz_mult=20, block_sz_mult=4):
    """ By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13,
    we can see that the 6th prime is 13. What is the 10001st prime number?
    answer: 104743
    """

    # array starts with 2
    count = 0
    i = 2
    start_val = 2
    # initial estimate of necessary array size
    max_n = n * start_sz_mult
    block_sz = n * block_sz_mult

    # True if prime
    is_prime = [True] * (max_n - start_val)
    while count < n:
        if is_prime[i - start_val]:
            # if we reach i and it is still True, it is prime
            count += 1
        for add in range(2 * i - start_val, max_n - start_val, i):
            # mark multiples of i as not prime
            is_prime[add] = False
        i += 1

        if i == max_n:
            # max_n is equivalent to corresponding int + 2
            start_val = max_n
            # no need to extend list, just update as needed
            is_prime = interval_prime_sieve(start_val, block_sz)
            max_n += block_sz
    return i - 1


def problem_8(n=13):
    """ largest product of 13 adjacent digits
        answer: 23514624000
     """
    input_str = "7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450"

    curr_prod = max_prod = 1
    # count should always track the number of values contributing to the prod
    count = 0
    l = []

    for index, d in enumerate(input_str):
        digit = int(d)

        # digit is zero, restart building up process
        if not digit:
            count = 0
            curr_prod = 1
        else:
            # multiply by next digit
            curr_prod *= digit
            count += 1

            # if count too long divide by first digit
            if count > n:
                curr_prod /= int(input_str[index - n])
                count -= 1
        # compare with maximum, reset if necessary
        if max_prod < curr_prod:
            max_prod = curr_prod

    return max_prod


def max_prod_finder(n, grid, start_i, start_j, next_i, next_j):
    """ find largest product in grid along axis defined by start + increment
        e.g. main diagonal would be start_i = 0, j = 0, next_i = 1, next_j = 1
        can rewrite problem 8 as: max_prod_finder(13, [input_str], 0, 0, 0, 1)
        after converting input_str to a list of ints
     """
    curr_prod = max_prod = 1
    # count should always track the number of values contributing to the prod
    count = 0
    l = []
    num_rows = len(grid)
    num_cols = len(grid[0])

    i = start_i
    j = start_j

    # for computing the first multiplied prod
    off_i = n * next_i
    off_j = n * next_j

    # can increment either one by zero and by negative nums, so check both
    while i < num_rows and j < num_cols and i >= 0 and j >= 0:
        digit = grid[i][j]
        print(digit)

        # digit is zero, restart count
        if not digit:
            count = 0
            curr_prod = 1

        else:
            # multiply product by the digit
            curr_prod *= digit
            count += 1

            # if too many nums included in prod, divide by first digit
            if count > n:
                curr_prod /= grid[i - off_i][j - off_j]
                count -= 1

        # compare with maximum, reset if necessary
        if max_prod < curr_prod:
            max_prod = curr_prod

        i += next_i
        j += next_j

    return max_prod


def problem_9(n=1000):
    """ pythagorean triplets a^2 + b^2 = c^2 s.t. a + b + c = 1000
        answer = 31875000
    """
    c_lim = n / 3

    # sqrs dict --> provides decent speed-up
    sqrs = {i: i ** 2 for i in range(1, n)}

    for c in range(n, c_lim, -1):
        for b in range(c, 1, -1):
            a = n - c - b
            # a should be <= b or else cases are repeated
            if a > b or a < 1:
                continue
            elif sqrs[c] - sqrs[b] == sqrs[a]:
                return a * b * c
    return -1


def problem_10(n=2000000):
    """ The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
        Find the sum of all the primes below two million.
        answer = 142913828922
    """
    count = 0
    sieve = [True] * n

    for i, is_prime in enumerate(sieve):
        if i < 2:
            sieve[i] = False
            continue
        if is_prime:
            count += i
        for add in range(2 * i, n, i):
            sieve[add] = False
    return count


def problem_11(n=4):
    """ What is the greatest product of four adjacent numbers in the same
        direction (up, down, left, right, or diagonally) in the 20x20 grid?
        answer: 70600674
    """
    raw_inp = """08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
        49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
        81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
        52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
        22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
        24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
        32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
        67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
        24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
        21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
        78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
        16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
        86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
        19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
        04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
        88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
        04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
        20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
        20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
        01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48"""

    # format grid as a list of lists
    grid = []
    for row in raw_inp.split("\n"):
        l = []
        # remove spaces, convert to ints
        for num in row.split(" "):
            if num:
                l.append(int(num))
        grid.append(l)

    num_rows = len(grid)
    num_cols = len(grid[0])

    # processed 20x20 grid correctly, grid is square
    assert num_rows == 20
    assert num_rows == num_cols
    # check every col
    for r in grid:
        assert len(r) == num_cols

    max_prod = 0

    # check all axes for maxes
    for x in range(num_rows):
        args = [
            (x, 0, 0, 1),  # horizontal
            (0, x, 1, 0),  # vertical
            (0, x, 1, 1),  # diagonal top to down right
            (x, 0, 1, 1),  # diagonal left side to down right
            (0, x, 1, -1),  # diagonal top to down left
            (x, num_rows - 1, 1, -1),  # diagonal right side to down left
        ]

        # check if each line's max is greater than the overall max, update if so

        for i, j, next_i, next_j in args:
            curr_prod = max_prod_finder(n, grid, i, j, next_i, next_j)
            if curr_prod > max_prod:
                max_prod = curr_prod

    return max_prod


def problem_12_naive(divs=500):
    """ We can see that 28 is the first triangle number to have over 5 divisors.
        What is the value of the first triangle number to have over 500 divisors?
        answer: 76576500
    """
    limit = 103672800

    sieve = num_divisors_sieve(limit)
    print("sieve complete")
    num_divs = 0

    i = 720720
    # i = 720720 * 7 # safer min bound

    # compute i - 1 tri num as initial
    t = 1
    tri_num = 0

    while num_divs < divs and t < limit:
        # compute triangular num
        tri_num += t
        num_divs = sieve[tri_num - 1]
        t += 1
        if t % 100000 == 0:
            print(tri_num)
    return tri_num


def tri_sieve(tri_lim=120000000):
    i = 0
    tri_sum = i
    tri_nums = []

    while tri_sum < tri_lim:
        i += 1
        tri_sum += i
        tri_nums.append(tri_sum)

    tri_divs = [0] * len(tri_nums)
    max_tri = tri_sum - i
    print("first lists done")
    min_val = max_tri
    for x in range(1, max_tri + 1):
        multiple = x
        while multiple < max_tri:
            for ind, tri in enumerate(tri_nums):
                while multiple <= tri:
                    if multiple == tri:
                        tri_divs[ind] += 1
                    multiple += x
                if tri_divs[ind] >= 500:
                    return tri_nums[ind]
                    if tri_nums[ind] < min_val:
                        min_val = tri_nums[ind]
    print(min_val)
    return tri_nums, tri_divs


def problem_12(divs=500):
    # super fast!!!
    lim = 1000000
    prime_sieve = is_prime_sieve(lim)
    prime_lst = [ind for ind, is_p in enumerate(prime_sieve) if is_p]

    t = 1
    tri_num = 0

    num_divs = 0

    while num_divs < divs:
        tri_num += t
        num_divs = num_divs_pf(tri_num, prime_lst)
        t += 1
        if t % 100000 == 0:
            print(tri_num)
    return tri_num


def problem_13(in_file="problem13.txt", num_digits=50, n=100):
    """
    sum the 100 50-digit numbers in text file problem13.txt
    print out the first 10 digits
    answer: 5537376230
    """

    # 100 50 digit numbers, invert sum matrix (sum by digit)
    sum_matrix = [[0] * n for _ in range(num_digits)]

    with open(in_file, "rb") as f:
        for i, line in enumerate(f):
            line = line.strip()

            for j in range(num_digits):
                # note that left-most (highest order) digit is index 0
                sum_matrix[j][i] = int(line[j])

    for digit in range(num_digits - 1, 0, -1):
        # sum the bottom row
        s = sum(sum_matrix[digit])
        # seperated into vars for readability
        u = s % 10
        tens = s / 10
        sum_matrix[digit - 1].append(tens)
        sum_matrix[digit] = u

    # last iteration just set index to sum
    sum_matrix[0] = sum(sum_matrix[0])
    int_str = "".join(str(x) for x in sum_matrix)

    print("first ten:", int_str[:10])

    return int_str


def collatz_naive(c_dict, n):
    """
    store already computed values in c_dict
    """
    chain_len = 1
    curr_val = n

    while curr_val > 1:
        if curr_val % 2 == 0:
            curr_val /= 2
        else:
            curr_val = 3 * curr_val + 1

        chain_len += 1

    return chain_len


# better should store a lookup table for lengths, build up


def collatz_rec(c_dict, n):
    if n in c_dict:
        # stored computation, and building back up recursively
        return c_dict[n]
    elif n == 1:
        # base case, necessary when dict is empty
        c_dict[n] = 1
        return 1
    else:
        # compute next step in the collatz series
        if n % 2 == 0:
            curr_val = n / 2
        else:
            curr_val = 3 * n + 1
        # store recursive computation in the dict
        c_dict[n] = 1 + collatz_rec(c_dict, curr_val)
        # return value which must be in dict
        return c_dict[n]


def problem_14(max_start=1000000):
    """
    Collatz sequences:
    n --> n/2 (n is even)
    n --> 3n + 1 (n is odd)

    Find longest chain with start_n < 1 million
    """

    max_chain = 1
    max_i = 1
    coll_dict = dict()

    # has to be odd
    for i in range(1, max_start):
        val = collatz_rec(coll_dict, i)
        if val > max_chain:
            max_chain = val
            max_i = i
    return max_i, max_chain, coll_dict


def problem_15(n=20):
    """
    number of paths between diagonal points on a grid can be represented with
    down moves as n 1s and n 0s. Answer is then number of permutations of 
    strings length 2n with n 1s and n 0s, e.g. (2n)! / 2*(n!)
    """
    fact_2n = range(n + 1, 2 * n + 1, 1)
    fact_n = range(1, n + 1)

    # keep it smol
    div = 1.0
    for f2n, f1n in zip(fact_2n, fact_n):
        div *= float(f2n)
        div /= float(f1n)

    return div


def problem_16(n=2, exp=1000):
    """
    2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
    What is the sum of the digits of the number 2^1000?
    Answer: 1366
    """
    # one liner: reduce(lambda a,b: int(a) + int(b), str(n))
    # multiplication out by hand
    m = [1]
    for mult in range(exp):
        for i in range(len(m)):
            m[i] *= n
        carry = 0
        for i in range(len(m) - 1, -1, -1):
            # carry
            if carry:
                m[i] += carry

            val = m[i]

            if val > 9:
                if i == 0:
                    # this is shitty...rewrite in reverse order and reverse at the end?
                    # front = map(int, str(val))     # python 2.7
                    front = [int(digit) for digit in str(val)]
                    m = front + m[1:]
                else:
                    m[i] = val % 10
                    carry = int(val / 10)

            else:
                carry = 0
    return sum(m)  # m, sum(m)


def problem_17(n=1001):
    """
    Letter counts if you sum up all words used up to 1000 (INCLUSIVE).
    Note, 342 written three hundred and forty two
    """
    unit_lst = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    teen_lst = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens_lst = [
        "",
        "ten",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    # create dictionary mappings with relevant lists
    unit_letter_map = {k: len(v) for k, v in enumerate(unit_lst)}
    teen_letter_map = {k: len(v) for k, v in enumerate(teen_lst)}
    tens_letter_map = {k: len(v) for k, v in enumerate(tens_lst)}

    HUNDRED = len("hundred")
    THOUSAND = len("thousand")
    H_AND = len("and")

    total_sum = 0

    for i in range(1, n):
        unit_i = i % 10
        tens_i = i / 10 % 10
        hundreds_i = i / 100 % 10
        thousands_i = i / 1000

        if thousands_i:
            # NOTE --> ONLY WORKS FOR <10,000; for more would have to extract this
            # for <100 into general helper func, then call for num + thousand, etc.
            total_sum += unit_letter_map[thousands_i]
            total_sum += THOUSAND

        if hundreds_i:
            total_sum += unit_letter_map[hundreds_i]
            total_sum += HUNDRED

            # add ' and '
            if tens_i or unit_i:
                total_sum += H_AND

        if tens_i == 1:
            # teen
            total_sum += teen_letter_map[unit_i]
        else:
            total_sum += tens_letter_map[tens_i]
            total_sum += unit_letter_map[unit_i]

    return total_sum


def problem_18(in_file="problem18.txt"):
    """
    By starting at the top of the triangle below and moving to adjacent numbers
    on the row below, the maximum total from top to bottom is 23.
                (corresponding ind)
                                        0 (source)
       3           0                       1
      7 4         1 2                     2 3
     2 4 6       3 4 5                   4 5 6
    8 5 9 3     6 7 8 9                 7 8 9 10
                10

                   0
                  0 3
                 4 2 0 
                1 4 0 6

    That is, 3 + 7 + 4 + 9 = 23.

    Find the maximum total from top to bottom of the triangle below:

    75
    95 64
    17 47 82
    18 35 87 10
    20 04 82 47 65
    19 01 23 75 03 34
    88 02 77 73 07 63 67
    99 65 04 28 06 16 70 92
    41 41 26 56 83 40 80 70 33
    41 48 72 33 47 32 37 16 94 29
    53 71 44 65 25 43 91 52 97 51 14
    70 11 33 28 77 73 17 78 39 68 17 57
    91 71 52 38 17 14 91 43 58 50 27 29 48
    63 66 04 68 89 53 67 30 73 16 69 87 40 31
    04 62 98 27 23 09 70 98 73 93 38 53 60 04 23

    # --> can't run dijkstra because "longest" path in a sense
    """
    import heapq as hq

    deb_map = {0: 3, 1: 7, 2: 4, 3: 2, 4: 4, 5: 6, 6: 8, 7: 5, 8: 9, 9: 3, 10: 0}
    grid = []
    # process input
    with open(in_file, "rb") as f:
        for line in f:
            grid.append([int(x) for x in line.split(" ")])

    # create adjacency list (n1, n2, weight)
    adj_lst = {}
    id_map = {}
    # adj_lst[0] = [(1, grid[0][0])]
    node_id = 0
    last_level = len(grid) - 1
    for level, lst in enumerate(grid):
        num_level_nodes = len(lst)

        first_below = node_id + num_level_nodes
        # compute differentials per line with max value edge weight as 0,
        # other edge weights are absolute value distance from max value

        max_val = max(lst)

        for offest_i, weight in enumerate(lst):
            differential = max_val - weight
            if level != last_level:
                # below_node = node_id + num_level_nodes
                below_node = first_below + offest_i
                diag_node = below_node + 1
                edges = [(differential, below_node), (differential, diag_node)]
            else:
                # all should go to the sink node
                edges = [(differential, first_below)]

            adj_lst[node_id] = edges
            # construct node_id --> value mapping
            id_map[node_id] = weight
            node_id += 1

    # sink node is created and has no real value
    id_map[first_below] = 0

    # run dijkstra, but need to track path
    visited = set()
    unexplored = []
    hq.heappush(unexplored, (0, 0, 0, [0]))

    while unexplored:
        path_len, from_node, to_node, path = hq.heappop(unexplored)

        # first_below = last node in graph, aka sink
        if to_node == first_below:
            # reconstruct path value --> this is unreadable but funsies!
            return reduce(lambda x, y: x + y, map(lambda ind: id_map[ind], path))

        visited.add(from_node)

        # add to unexplored if hasn't been searched already
        for w, neighbor in adj_lst[to_node]:
            if neighbor not in visited:
                hq.heappush(
                    unexplored, (w + path_len, to_node, neighbor, path + [neighbor])
                )

    return -1


def problem_19(start_date=(1, 1, 1901), end_date=(31, 12, 2000)):
    """ 1 Jan 1900 was a Monday.
        Thirty days has September,
        April, June and November.
        All the rest have thirty-one,
        Saving February alone,
        Which has twenty-eight, rain or shine.
        And on leap years, twenty-nine.
        A leap year occurs on any year evenly divisible by 4, 
        but not on a century unless it is divisible by 400.
    """
    # difference in start day per month
    month_days = [x % 7 for x in [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
    suns = 0
    # start day is Monday
    day_count = 1

    for year in range(1900, end_date[2] + 1):
        for month, month_diff in enumerate(month_days):
            day_count += month_diff
            # february leap year
            if month == 1 and year % 4 == 0:
                # century
                if year % 400 == 0 or year % 100 > 0:
                    day_count += 1
            day_count %= 7
            # sunday start day!
            if day_count == 0:
                # only count if after start date
                if year >= start_date[2]:
                    suns += 1
    return suns


def problem_20(n=100):
    """
    sum digits in n!
    both naive and space conservative work
    answer 648
    """
    # not space efficient
    n_fact = 1
    for i in range(1, n + 1):
        n_fact *= i

    # insights: we can divide all the zeroes out as we go
    # while n_fact % 10 == 0:
    #     n_fact /= 10
    return sum([int(x) for x in str(n_fact)])


def problem_21(n=10000, tab_size=3):
    """
    sum of amicable numbers below 10000
    answer: 31626
    """
    # create proper divisors table
    d = divisors_tab(tab_size * n, proper=True)

    tot_sum = 0

    for i in range(1, n):
        d_i = d[i]
        # check once per pair
        if i < d[i]:
            if d_i > tab_size * n:
                # increase tab size
                print("too big, ", d_i)
                # tbh this is hacky, can also just put in a manual divisors test
                # for the few numbers that are greater than 10000 but whatevs
            elif d[d_i] == i:
                tot_sum += d_i + i
    return tot_sum


def problem_22(in_file="problem22.txt", qsort=False):
    """
    sort + score names
    answer: 871198282
    """
    # quicksort about twice as slow as built-in sort
    # def quick_sort(lst):
    # if len(lst) <= 1:
    #     return lst

    # pivot = lst[0]
    # left = quick_sort([elt for elt in lst[1:] if elt < pivot])
    # right = quick_sort([elt for elt in lst[1:] if elt >= pivot])
    # return left + [pivot] + right

    import re

    # read in names, clean & ensure uppercase
    name_lst = []

    with open(in_file, "rb") as f:
        raw_inp = f.read()
        pattern = re.compile("\W")
        name_lst = [re.sub(pattern, "", name.upper()) for name in raw_inp.split(",")]

    # sort list
    # if qsort:
    #     name_lst = quick_sort(name_lst)
    # else:

    name_lst.sort()

    # # compute scores per name
    char_offset = ord("A") - 1
    total_score = 0
    for i, name in enumerate(name_lst):
        score = 0
        for c in name:
            score += ord(c) - char_offset
        total_score += score * (i + 1)

    return total_score


def problem_23(lim=28123):
    """
    A number n is called deficient if the sum of its proper divisors is less 
    than n and it is called abundant if this sum exceeds n. Find the sum of all
    the positive integers which cannot be written as the sum of two abundant numbers.
    answer: 4179871

    """
    # create proper divisors table up to size limit
    tab = divisors_tab(lim, proper=True)

    # list of abundant nums
    abundant_nums = []
    for i, divs_i in enumerate(tab):
        if divs_i > i:
            abundant_nums.append(i)

    ab_sums = set()
    for i in abundant_nums:
        for j in abundant_nums:
            s_ij = i + j
            if s_ij < lim:
                ab_sums.add(s_ij)

    return sum(range(lim)) - sum(ab_sums)


def problem_24(n_perm=1000000, digs=10):
    """
    A permutation is an ordered arrangement of objects. 
    The lexicographic permutations of 0, 1 and 2 are:
    012   021   102   120   201   210
    What is the millionth lexicographic permutation of the digits 0-9?
    Procedure as follows, subtracting factorials to figure out the digit:

    # numbers that begin with 0: 9! = 362880
    # numbers that begin with 1: 9! = 362880 (total 725760)
    # numbers that begin with 2: 9! = 362880 (total 1,088,640)
      ==> so num begins with 2 [201345789 = number 725761]
      ==> need 1000000 - 725760 = 274240
            (dig 2: 8! = 40320, max ceiling = 8! * 6 under, so digit = 7 (0, 1, 3, 4, 5, 6 done))
            ==> 27 _ _ _ _ _ _ _ _ still need: 274240 - 8! * 6 = 32350
            ...
    """
    remaining_digits = list(range(digs))
    remainder = n_perm

    num = ""

    # outer loop = selecting digits:
    for digit in range(digs - 1):
        fact = factorial(digs - 1 - digit)
        num_subtracts = 0
        while remainder > 0:
            remainder -= fact
            num_subtracts += 1

        # reflect how many actually left
        remainder += fact
        num += str(remaining_digits[num_subtracts - 1])
        del remaining_digits[num_subtracts - 1]

    # one digit left
    num += str(remaining_digits[0])
    return num


def sub_strings(s, d, parts={}):
    """
    Compute all valid partitions of s given a dictionary d of valid words.
    Call function directly or use wrapper below in order to pass in a global
    partitions dictionary and debug.
    Overall, all computation is stored and should never repeat a computation.
    """
    # base case: check if the entire string is a valid word
    if s not in parts:
        parts[s] = [s] if s in d else []

    for i in range(len(s)):
        # check if s[:i] is a valid word
        if s[:i] in d:

            # already computed partitions on substring s[i:]
            if s[i:] in parts:
                rem_partitions = parts[s[i:]]

            # recursively call function on remainder of substring
            else:
                rem_partitions = sub_strings(s[i:], d, parts)
            # partitions on s should include front half (s[i:])
            # with all valid partitions of remaining string
            parts[s] += [s[:i] + " " + sub for sub in rem_partitions]
    return parts[s]


def str_partitions(s, d):
    # parts = {} # for debugging
    # return parts
    return sub_strings(s, d)


def problem_25(digits=1000):
    """
    index of first fibonacci term with >=1000 digits
    """
    # compute fib terms as we go and also keep running sum, so one pass
    f_1 = 0
    f_2 = 1
    # tracks index of f_2
    n = 1
    even_sum = 0
    limit = 10 ** (digits - 1)
    while f_2 < limit:
        # shift previous computed term down
        f_1, f_2 = f_2, f_1 + f_2
        # compute new term
        n += 1
    return n


def problem_26(lim=1000):
    """ longest recurring cycle in decimal fractions"""

    def div_cycle(n, print_cyc=False):
        """ helper function to determine the length of the cycle in 1/n.
        Essentially, step through long division and once we reach a remainder
        that has already been seen, we have found the cycle (between the
        two remainders) """
        base = 10
        remainder = 1

        # for debugging
        cyc = ""

        # track remainders and length of cycle before then
        rems = {}
        cycle_len = 0

        while remainder:
            # for debugging
            dec = remainder * base / n
            # long-division "carry"
            remainder *= base
            # already seen remainder is the start of next cycle
            if remainder in rems:
                if print_cyc:
                    print("cycle found: ", cyc[rems[remainder] :])
                return cycle_len - rems[remainder]
            # store the index at occurence of remainder, because there can be
            # digits before the cycle begins, need to subtract out
            rems[remainder] = cycle_len

            cycle_len += 1
            remainder %= n
            cyc += str(dec)
        return -1

    max_len = 0
    max_n = 0
    # loop over all numbers and find the n with highest corresponding cycle len.
    for num in range(1, lim):
        val = div_cycle(num)
        if val > max_len:
            max_len = val
            max_n = num
    return max_n


def problem_27(p_ceil=120000):
    """ consecutive quadratic primes: n^2 + an + b = prime number """
    # b must be prime > | 2 | (n = 0)
    # a must be odd
    b_lim = 1000
    a_lim = 999
    max_a = max_b = max_consec = 0

    # create a prime sieve, p_ceil likely < 100: 100^2 + 999*100 + ~1000 = 110900
    prime_sieve = is_prime_sieve(p_ceil)
    primes = {i for i, p in enumerate(prime_sieve) if p}
    lim_primes = []

    # b needs to include negative primes up to |1000|
    for p in primes:
        if p > b_lim:
            continue
        lim_primes.append(p)
        lim_primes.append(-p)

    # loop over all possible a's and b's
    for b in lim_primes:
        for a in range(-a_lim, a_lim, 2):
            consec = True
            n = 0
            while consec:
                p = n * (n + a) + b
                consec = p in primes or -p in primes
                n += 1

            if n > max_consec:
                max_consec = n
                max_a, max_b = a, b
    print(max_a, max_b, max_consec)
    return max_a * max_b


def problem_28(side_len=1001):
    # sum follows pattern: 1 + (1 + 2 = 3) + (3 + 2 = 5) + (5 + 2 = 7) + (7 + 2 = 9) + (9 + 4 = 12)..
    s = i = 1
    step = 2
    count = 4
    # values in 1001 x 1001 square
    lim = side_len ** 2
    while i < lim:
        # number of times to stay at one step
        i += step
        s += i
        count -= 1
        if count == 0:
            step += 2
            count = 4
    return s


def problem_29(lim=100):
    """ a^b 2<=a,b<=100, number of distinct vals for all possible a,b """
    # derp...
    derp = set()
    for a in range(2, lim + 1):
        for b in range(2, lim + 1):
            derp.add(a ** b)
    return len(derp)


def problem_30(lim=1000000):
    """ find all the numbers where the sum of the digit d^5 = the number
        e.g. 4150 
        limit set to 7 digits bc 9**5 (9 as a digit) 7 times is still 6 digits"""
    # don't keep redoing digit computation
    d_to_fifth = {str(x): x ** 5 for x in range(10)}

    s = 0
    for n in range(2, lim):
        dig_sum = 0
        for dig in str(n):
            dig_sum += d_to_fifth[dig]
        if dig_sum == n:
            print(n)
            s += n
    return s


def problem_31(target=200, coins=[1, 2, 5, 10, 20, 50, 100, 200]):
    """ coin counting problem: how many ways to make 200p with coins of following
        values: 1, 2, 5, 10, 20, 50, 100, 200
        answer: 73682
    """
    sums_mat = [[0] * len(coins) for _ in range(target)]
    coin_vals = {x: c for x, c in zip(range(len(coins)), coins)}

    # init each coin to it's value
    for col_ind, i in enumerate(coins):
        sums_mat[i - 1][col_ind] = 1

    # iterate over all entries in the matrix
    for i in range(target):
        for j in range(len(coins)):
            # check if it's possible to get this value with lead value j
            remaining = i - coin_vals[j]
            if remaining >= 0:
                # sum all ways to make remaining value (after subtracting val j)
                # up to max lead value of j
                sums_mat[i][j] = sum(sums_mat[remaining][: j + 1])
    return sums_mat


def problem_32():
    """ 1 to 9 pandigital sums, find all products with this """
    # total digits in factors sums to 5
    prods = set()
    # a x b = c
    for a in range(123, 9877):
        a_str = str(a)
        a_digs = len(a_str)
        for b in range(1, 99):
            b_str = str(b)
            digs = set(b_str)
            digs.update(a_str)
            # total digits in a and b need to sum to 5
            if a_digs + len(b_str) != 5 and len(digs) != 5:
                continue
            c = a * b
            digs.update(str(c))
            # c must have 4 digits
            if len(digs) == 9 and ("0" not in digs) and len(str(c)) == 4:
                prods.add(c)

    return sum(prods)


def problem_33():
    """ digit cancelling fractions: e.g. 49/98 == 4/8
        for all two digit numerators and denominators
        check if all 4 combinations of digits: ab/cd --> a/c, a/d, b/c, b/d == ab/cd
        if one does, return the x/y pair that is == to ab/cd
        corner case if d is 0, don't test this """
    frac_lst = []
    for denom in range(10, 100):
        for num in range(10, denom):
            dec_val = num * 1.0 / denom
            # fraction as ab/cd
            a = num / 10
            b = num % 10
            c = denom / 10
            d = denom % 10
            if a != c and a != d and b != c and b != d:
                continue

            # need to find out the val of resulting fraction when the two same digits are "cancelled"
            pair = (0, 0)
            nums = [a, b]
            denomes = [c, d]
            for i_n, n in enumerate(nums):
                for i_d, d in enumerate(denoms):
                    if not (n and d):  # either n or d are zero
                        break
                    if n == d:
                        pair = (nums[1 - i_n], denoms[1 - i_d])
            # make sure no 0 division
            if pair[0] == 0 or pair[1] == 0:
                continue

            # if pair[0] * 1./pair[1] == dec_val:  # if a/b = c/d ==> a*d = b*c
            if pair[0] * denom == pair[1] * num:
                frac_lst.append((pair))

    num = 1
    denom = 1
    for n, d in frac_lst:
        num *= n
        denom *= d

    return frac_lst, (num, denom)


def problem_34(lim=2600000):
    """ curious numbers: factorial of digits in a number sum to number itself
        max limit because 9! * 7 < 2600000 (so no 8-dig. num can be "curious")"""
    facts = {digit: factorial(digit) for digit in range(10)}
    num_sum = 0
    # exclude 1 digit numbers
    for num in range(10, lim):
        target = num
        dig_sum = 0
        while num and dig_sum <= target:
            dig_sum += facts[num % 10]
            num /= 10
        if dig_sum == target:
            print("Target reached: ", target)
            num_sum += target

    return num_sum


def problem_35(lim=1000000):
    """ circular primes below one million """
    # helper functions to rotate a number -- all integer is much faster
    def rotate_num_str(n):
        str_n = str(n)
        rotations = []
        for i in range(len(str_n)):
            rotations.append(str_n[i:] + str_n[:i])
        return [int(r) for r in rotations]

    def rotate_num(n):
        # much faster
        num = n
        rotations = []
        num_digits = len(str(n))
        for i in range(num_digits):
            rotate_dig = num % 10
            # if
            num /= 10
            num += rotate_dig * 10 ** (num_digits - 1)
            rotations.append(num)
        return rotations

    def has_even_digit(n):
        num = n
        while num > 0:
            digit = num % 10
            if digit % 2 == 0:
                return True
            num /= 10
        return False

    # no numbers with an even digit can be a rotating prime
    primes = [
        i
        for i, p in enumerate(is_prime_sieve(lim))
        if p and not (has_even_digit(i) and len(str(i)) > 1)
    ]
    circular_primes = set()
    non_circ = set()

    for i in primes:
        if i in circular_primes or i in non_circ:
            continue
        rotations = rotate_num(i)
        circ = True
        for r in rotations:
            if r not in primes:
                circ = False
                for r in rotations:
                    non_circ.add(r)
                non_circ.update(rotations)
                break
        if circ:
            for r in rotations:
                circular_primes.add(r)
            # circular_primes.update(rotations)
    return circular_primes


def problem_36(lim=1000000):
    """The decimal number, 585 = 1001001001_2 (binary), is palindromic in both bases.
        Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.
        (Please note that the palindromic number, in either base, may not include leading zeros.)"""
    # constraints:
    # must be odd (no leading zeros, base 2 --> ends in a 1); first digit must also be odd
    # first check if palindrome in base 10 (much shorter)
    pows_of_2 = []
    i = 1
    while i <= lim:
        pows_of_2.append(i)
        i *= 2

    pows_of_2.reverse()

    # store the powers of 2 that we have already computed
    converted = dict()

    total_sum = 0
    pals = set()
    for i in range(1, lim + 1, 2):
        if is_palindrome(i):
            i_b2 = int(
                "".join(str(x) for x in convert_to_base_2(i, converted, pows_of_2))
            )
            if is_palindrome(i_b2):
                pals.add(i)
    return pals, sum(pals)


def convert_to_base_2(n, lookup, pows_of_2):
    remainder = n
    i = 0
    base_2 = []
    for i, pow_of_2 in enumerate(pows_of_2):
        if remainder in lookup:
            base_2 += lookup[remainder][i:]
            break
        else:
            if remainder - pow_of_2 >= 0:
                remainder -= pow_of_2
                base_2.append(1)
            else:
                base_2.append(0)
    # num = int("".join(str(x) for x in base_2))
    lookup[n] = base_2  # num
    return base_2


def l_truncate_prime(n, is_prime_sieve):
    while n:
        if not is_prime_sieve[n]:
            # not prime any more:
            return False
        n %= 10**(len(str(n)) - 1)
    return True


def r_truncate_prime(n, is_prime_sieve):
    while n:
        if not is_prime_sieve[n]:
            # not prime any more:
            return False
        n //= 10
    return True


def problem_37(lim=1000000):
    sieve = is_prime_sieve(lim)
    primes = [i for is_prime, i in zip(sieve, range(lim)) if is_prime]
    successes = list()
    for i in primes:
        if l_truncate_prime(i, sieve) and r_truncate_prime(i, sieve):
            successes.append(i)
    return successes

    # start with prime val
    # remove digits from left and check prime
    # remove digits from right and check prime

    # truncatable primes; find 11 of them
    # [23, 37, 53, 73, 797, 3797]
    pass



def problem_38():
    # since multiplied by 1 first, n must start with 9 (9, 90, ...) -- needs to be greater than 918273645
    # >= 92
    search_range = list(range(91, 97)) + list(range(9123, 9877))
    max_i = 0
    max_pandig = ""
    for i in search_range:
        x = get_concats(i)
        set_x = set(x)
        set_x.add('0')
        if len(set_x) == 10:
            print(i, ", ", x)
            if i > max_i:
                max_i = i
                max_pandig = x
    return max_i, max_pandig


def get_concats(n, max_tries=10):
    curr_prod = ""
    i = 1
    while len(curr_prod) < 9 and max_tries:
        curr_prod += str(i * n)
        max_tries -= 1
        i += 1
    return curr_prod


#############################################################################
############################## FEELIN CRAZY #################################
#############################################################################


def problem_230(lim_digit=17):
    def get_first_long_enough(k, fib_dict, len_A):
        order = len(str(k // len_A)) - 1
        lst = fib_dict[order]
        # return first fib >= k
        for n in lst:
            if k <= n * len_A:
                return n
        return fib_dict[order + 1][0]

    def prep_fib_dicts(digs=18):
        """ Helper to create fibonacci dictionary nested by digit """
        from collections import defaultdict

        f_1 = 1
        old_f_1 = "1a"
        f_2 = 2
        old_f_2 = "1b"
        nums = defaultdict(list)

        order = 0
        nums[order] = [1]
        fib_pairs = dict()
        fib_pairs[old_f_1] = ("A", None)
        fib_pairs[old_f_2] = ("B", None)

        while order < digs:
            nums[order].append(f_2)
            fib_pairs[f_2] = (old_f_1, old_f_2)
            # shift previous computed term down, use swap syntax
            old_f_1, old_f_2 = old_f_2, f_2
            f_1, f_2 = f_2, f_1 + f_2
            fib_pairs[f_2] = (old_f_1, old_f_2)
            order = len(str(f_2)) - 1
            # compute new term
        return nums, fib_pairs

    def a_or_b(d, fib_dict, fib_pairs, len_A):
        # len_A = 100
        # fib_dict, fib_pairs = prep_fib_dicts(digs=10)

        fib_num = get_first_long_enough(d, fib_dict, len_A)
        print("Closest_fib: ", fib_num)
        fib_left, fib_right = fib_pairs[fib_num]

        d_mod = d % len_A
        d_floor = d // len_A
        # backtrace through the pairs of digits, choosing left and right as fit
        while fib_right is not None:
            val = fib_left if isinstance(fib_left, int) else 1
            d_floor -= val
            if d_floor >= 0:
                fib_left, fib_right = fib_pairs[fib_right]
            else:
                d_floor += val
                fib_left, fib_right = fib_pairs[fib_left]
        return fib_left, d_mod

    mappings = {
        "A": "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679",
        "B": "8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196",
    }

    len_A = len(mappings["A"])

    ns = list(range(lim_digit, -1, -1))
    ds = [(127 + 19 * n) * 7 ** n for n in ns]

    # note word lengths will be: len(word_i) = fib_i * 100
    fib_dict, fib_pairs = prep_fib_dicts(digs=len(str(ds[0] // len_A)) + 1)

    final_answer = ""
    for d in ds:
        print(f"Starting {d}")
        letter, d_mod = a_or_b(d, fib_dict, fib_pairs, len_A)
        final_answer += mappings[letter][d_mod - 1]
    return final_answer


def prob_230_brute(max_dig=10):
    # LOL
    mappings = {
        "A": "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679",
        "B": "8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196",
    }

    ns = list(range(max_dig))
    ds = [(127 + 19 * n) * 7 ** n for n in ns]
    print(ds)
    # note word lengths will be: len(word_i) = fib_i * 100

    # track the fib number and just the words that we care about at the moment
    # maybe store a placeholder value for the word itself
    # then when we index into the word (knowing that the appropriate digit is in this sequence, we index into either A or B)
    f_1 = 1
    f_2 = 1  # represents len(term_2) // 100
    fw_1 = "A"  # represents the sequence of var a, b that forms this word
    fw_2 = "B"

    d_digits = []

    for d in ds:
        print(f"Starting {d}")
        d_floor = d // 100

        while d_floor >= f_2:
            # compute new word term
            fw_1, fw_2 = fw_2, fw_1 + fw_2
            f_1, f_2 = f_2, f_1 + f_2

        # have to figure out how far back to index (last 100 digits won't always cut it)
        if d_floor < f_2:
            print("fib num: ", f_2)
            dig = int(mappings[fw_2[d_floor]][d % 100])
            print(dig)
            d_digits.append(dig)

    return d_digits
