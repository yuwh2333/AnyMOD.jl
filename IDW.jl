    function computeIDW(x_train::Vector{Dict}, y_train::Vector{Float64}, x::Dict, par::Int64)
        w = zeros(length(x_train))
        u_up = 0.0
        u_down = 0.0
        p = par
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
        if !isempty(dis)
            return y_train[argmin(dis)]
        else
            return 0
        end
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

    function L2NormDict(x1::Dict,x2::Dict)
        d = 0
        for (key,value) in x1
            if key in keys(x2)
                d += (value-x2[key])^2
            else
                d += (value)^2
            end
        end
        return sqrt(d)
    end

    function find_min_and_second_min_indices(arr)
        if length(arr) < 2
            error("Array must contain at least two elements.")
        end
    
        # Initialize minimum and second minimum
        if arr[1] < arr[2]
            min_index, second_min_index = 1, 2
        else
            min_index, second_min_index = 2, 1
        end
    
        for i in 3:length(arr)
            if arr[i] < arr[min_index]
                second_min_index = min_index
                min_index = i
            elseif arr[i] < arr[second_min_index]
                second_min_index = i
            end
        end
    
        return min_index, second_min_index
    end
    
    function computelinearextra(x_train::Vector{Dict},y_train::Vector{Float64}, x::Dict)
        L2_dis = []
        #compute l2norm to all training datas
        for i in 1:length(x_train)
            d = 0
            for (key,value) in x
                d += (value - x_train[i][key])^2
            end
            d = sqrt(d)
            push!(L2_dis,d)
        end
        L1_x = sum(values(x))
        #if x has the smallest L1-Norm, then do the extrapolation
        extra_boo = true
        for i in 1:length(x_train)
            if L1_x > sum(values(x_train[i]))
                extra_boo = false
            end
        end
        if extra_boo == true
            return computeNN(x_train, y_train,x)
        elseif length(y_train) == 1
                return y_train[1]
        else          
            n1,n2 = find_min_and_second_min_indices(L2_dis)
            return y_train[n2] + L2NormDict(x,x_train[n2])/L2NormDict(x_train[n1],x_train[n2])*(y_train[n1]-y_train[n2])
        end
    end




