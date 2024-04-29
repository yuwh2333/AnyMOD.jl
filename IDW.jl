
module IDW

    export computeIDW,computeNN
    using DataFrames
    function computeIDW(x_train::Vector{Dict}, y_train::Vector{Float64}, x::Dict)
        w = zeros(length(x_train))
        u_up = 0.0
        u_down = 0.0
        p = 40
        #u = 0.0
        for i in 1:length(x_train)
            d = 0
            for (key,value) in x
                d += (value - x_train[i][key])^2
            end
            d = sqrt(d)
            w[i] = 1 / (d^p)
            u_up += w[i] * y_train[i]
            u_down += w[i]
        end
        if u_down != 0
             return u_up / u_down
        else
            return 0
        end
    end


    function computeNN(x_train::Vector{Dict}, y_train::Vector{Float64},x::Dict)
        dis = []
        for i in 1:length(x_train)
            d = 0
            for (key,value) in x
                d += (value - x_train[i][key])^2
            end
            d = sqrt(d)
            push!(dis,d)
        end
        return y_train[argmax(dis)]
    end
end

