def slope_of_cubic(coefficients, x):
    a, b, c, d = coefficients 
    slope = 3 * a * x**2 + 2 * b * x + c  
    return slope
a = float(input("Enter coefficient for x^3 (a): "))
b = float(input("Enter coefficient for x^2 (b): "))
c = float(input("Enter coefficient for x (c): "))
d = float(input("Enter constant term (d): "))
cf=(a,b,c,d)
x=float(input("Enter the value of x at which to calculate the slope: "))
slope=slope_of_cubic(cf, x)
print(f"The slope of the polynomial at x is: {slope}")