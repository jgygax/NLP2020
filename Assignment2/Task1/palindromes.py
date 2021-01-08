import sys

class Palindrome():

    def find_longest_palindrome(self, x):
        self.resize_and_reset_mem(x)
        return self.find_palindromes(x, 0, len(x)-1)

    def resize_and_reset_mem(self, x):
        self.mem = [len(x)*[""] for i in range(len(x))]

    def find_palindromes(self, x, start, end):
        if start > end:
            return ''
        if self.mem[start][end] != '':
            return self.mem[start][end]

        longest_palindrome=x[start]
        for idx1 in range(start,end):
            for idx2 in range(end, idx1, -1):
                if x[idx1]==x[idx2]:
                    palindrome = x[idx1] + self.find_palindromes(x,idx1+1, idx2-1) + x[idx2]
                    if len(palindrome)>len(longest_palindrome):
                        longest_palindrome = palindrome
        self.mem[start][end]=longest_palindrome
        return longest_palindrome

if __name__ == '__main__':
    p = Palindrome()
    print(p.find_longest_palindrome(sys.argv[1]))
