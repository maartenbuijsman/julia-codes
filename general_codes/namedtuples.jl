i = 1
f = 3.14
s = "Julia"

my_quick_namedtuple = (; i, f, s)


pm = (; i, f, s)

pm2 = (; k=1, g=2, j="top")

pm4 = merge(pm,pm2,(ii = 1, ff = 3.14))


aa, bb = 3,-5.3

pm5 = merge(pm4,(; aa, bb))

typeof(pm5)


y(x,p) = p.f * x + p.g*x^2

y(5,pm5)