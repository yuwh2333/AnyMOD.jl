    function computeIDW(x_train::Vector{Dict}, y_train::Vector{Float64}, x::Dict)
        w = zeros(length(x_train))
        u_up = 0.0
        u_down = 0.0
        p = 40
        for i in 1:length(x_train)
            d = 0
            for (key,value) in x
                d += (value - x_train[i][key])^2
            end
            d = sqrt(d)
            if d!=0
                w[i] = 1 / (d^p)
            else
                return y_train[i]
            end
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
        return y_train[argmin(dis)]
    end

    function computedualIDW(x_train::Vector{Dict}, y_train::Vector{Float64}, x::Dict, dual_dic::Vector{Dict})
        w = zeros(length(x_train))
        u_up = 0.0
        u_down = 0.0
        p = 40
        for i in 1:length(x_train)
            d = 0
            for (key,value) in x
                temp_sym = Symbol(string(key), "<Dual")
                if haskey(dual_dic[i], temp_sym)
                    d += (value - x_train[i][key])^2*abs(dual_dic[i][temp_sym])#/abs(sum(values(dual_dic[i])))
                end
            end
            d = sqrt(d)
            if d!=0
                w[i] = 1 / (d^p)
            else
                w[i] = 0
            end
            u_up += w[i] * y_train[i]
            u_down += w[i]
        end
        if u_down != 0
             return u_up / u_down
        else
            return 0
        end
    end




