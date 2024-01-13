from TinyStatistician import TinyStatistician

a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))
# Output:
82.4
print(TinyStatistician().median(a))
# Output:
42.0
print(TinyStatistician().quartile(a))
# Output:
[10.0, 59.0]
print(TinyStatistician().percentile(a, 10))
# Output:
4.6
print(TinyStatistician().percentile(a, 15))
# Output:
6.4
print(TinyStatistician().percentile(a, 20))
# Output:
8.2
print(TinyStatistician().var(a))
# Output:
15349.3
print(TinyStatistician().std(a))
# Output:
123.89229193133849