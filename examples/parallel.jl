import Base.Threads.@spawn

function fib(n::Int)
    if n < 2
        return n
    end
    t = @spawn fib(n - 2)
    return fib(n - 1) + fetch(t)
end

Threads.@threads for i = 1:10
    println("i = $i on thread $(Threads.threadid())")
end

Threads.nthreads()


a = zeros(10)

# parallel loop
Threads.@threads for i = 1:10
    a[i] = Threads.threadid()
end
a