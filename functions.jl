#include("./src/decomposition/objects.jl")
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

function computenopola(x_train::Vector{Dict},y_train::Vector{Float64}, x::Dict)
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
    #if x has the smallest L1-Norm, then don't do interpolation
    no_boo = true
    for i in 1:length(x_train)
        if L1_x > sum(values(x_train[i]))
            no_boo = false
        end
    end
    if no_boo == false
        return computeNN(x_train, y_train,x)
    else          
        return y_train[length(y_train)]
    end
end


function computeSurrogates(benders_obj::bendersObj, surroSelect_sym::Symbol, input, par_df, subData)
   
    # get the estimated cost from top-problem (must be before running top problem again without stabilization!)
	cutVar_df = copy(benders_obj.top.parts.obj.var[:cut]) 
	cutVar_df[!,:estCost] = value.(cutVar_df)[!,:var]
	select!(cutVar_df,Not([:var]))
    cutVar_df[!,:timeSub].=Millisecond(0)
    cutVar_df[!,:sur].=0.0    
    #compute surrogates
    if benders_obj.itr.cnt.i>2
        for row in eachrow(cutVar_df)
            if surroSelect_sym == :IDW_cc || surroSelect_sym == :IDW_simu
                row.sur = computeIDW(subData[(row.Ts_dis, row.scr)].x, subData[(row.Ts_dis, row.scr)].z, input, par_df[id_int,:p])
            end
            if surroSelect_sym == :NN_cc || surroSelect_sym == :NN_simu
                row.sur = computeNN(subData[(row.Ts_dis, row.scr)].x, subData[(row.Ts_dis, row.scr)].z, input)
            end
            if surroSelect_sym == :extra || surroSelect_sym == :extra_simu
                row.sur = computelinearextra(subData[(row.Ts_dis, row.scr)].x, subData[(row.Ts_dis, row.scr)].z, input)
            end
            if surroSelect_sym == :nopola
                row.sur = computenopola(subData[(row.Ts_dis, row.scr)].x, subData[(row.Ts_dis, row.scr)].z, input)
            end
        end        
    end
    cutVar_df[!,:diff] = cutVar_df[!,:sur]  .- cutVar_df[!,:estCost] 
    insertcols!(cutVar_df, :actCost => Vector{Union{Nothing, Float64}}(nothing, nrow(cutVar_df)))
    surroNeg_boo = false
    for row in eachrow(cutVar_df) 
        if row.diff <-1 
            row.diff = 1
            surroNeg_boo = true
        end
    end 

    # define maxTup
    sMaxDiff_tup = tuple((cutVar_df[findall(maximum(cutVar_df[!,:diff]) .== cutVar_df[!,:diff]), :] |> (z -> map(x -> z[1,x], [:Ts_dis, :scr])))...)
    
    
    return sMaxDiff_tup, cutVar_df, surroNeg_boo
end

function resDatatoDict(resData_obj::resData)
    """ 
    input: resData_obj
    output: Dict_obj
    function: save the resData.capa into a dictionary form, which then serves as the input for surrogates

    """
    Dict_obj = Dict{Symbol, Float64}() 
    for sys in (:tech, :exc)
		for sSym in keys(resData_obj.capa[sys]), capaSym in keys(resData_obj.capa[sys][sSym])
            if sys == :tech
                for (index,row) in enumerate(eachrow(resData_obj.capa[sys][sSym][capaSym]))
                    var_name = Symbol(string(sys),"<", string(sSym),"<", string(capaSym),"<",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_expansion[index])
                    Dict_obj[var_name] = row.value
                end
            elseif sys == :exc
                for (index,row) in enumerate(eachrow(resData_obj.capa[sys][sSym][capaSym]))
                    var_name = Symbol(string(sys),"<",string(sSym),"<",string(capaSym),"<",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_from[index],"-",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_to[index])
                    Dict_obj[var_name] = row.value
                end
            end                 
		end		
	end
    return Dict_obj
end

function saveDual(cutData_dic, cut_group, resData_obj, benders_obj, dualvr)
    inner_dict = Dict{Symbol, Float64}()
    for (id,s) in enumerate(collect(keys(cutData_dic)))
        if s in cut_group
            for sys in (:tech, :exc)
                for sSym in keys(cutData_dic[s].capa[sys]), capaSym in keys(cutData_dic[s].capa[sys][sSym])
                    if sys == :tech
                        for (index,row) in enumerate(eachrow(cutData_dic[s].capa[sys][sSym][capaSym]))
                            var_name = Symbol(string(sys),"<", string(sSym),"<", string(capaSym),"<",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_expansion[index],"<Dual")
                            inner_dict[var_name] = row.dual
                        end
                    elseif sys == :exc
                        for (index,row) in enumerate(eachrow(cutData_dic[s].capa[sys][sSym][capaSym]))
                            var_name = Symbol(string(sys),"<",string(sSym),"<",string(capaSym),"<",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_from[index],"-",printObject(resData_obj.capa[sys][sSym][capaSym],benders_obj.top,rtnDf = (:csvDf,)).region_to[index],string("<Dual"))
                            inner_dict[var_name] = row.dual
                        end 
                    end 
                end                 
            end		
            if haskey(dualvr, s)
                push!(dualvr[s],inner_dict)
            else
                dualvr[s] = [inner_dict]
            end
        end
    end
end


mutable struct actStatus
    check_Conv :: Bool
    rtn_boo :: Bool
    last_stab_obj :: Float64
    real_benders_best_obj :: Float64
    fake_benders_best_obj :: Float64
    futworkers_dic :: Dict{Int64, Future}
    cutData_dic :: Dict{Tuple{Int64,Int64},resData}
	timeSub_dic :: Dict{Tuple{Int64,Int64},Millisecond}
	lss_dic :: Dict{Tuple{Int64,Int64},Float64}
    cutGroup :: Vector{Tuple{Int64,Int64}}
    function actStatus()
        new(true, false, Inf, Inf, Inf, Dict(),Dict(),Dict(),Dict(),[])
    end
end

mutable struct SubObj
    x:: Vector{Dict}
    z:: Vector{Float64}
    actItr:: Int64 #last iteration number when a result is fetched
    strItr:: Int64 #last iteration number when a SP started
    function SubObj()
        sub_obj = new()
        sub_obj.x = Vector{Dict}()
        sub_obj.z = Vector{Float64}()
        sub_obj.actItr = 0
        sub_obj.strItr = 0
        return sub_obj
    end
end

function savePoint!(SubData, input, cutData_dic, s, benders_obj)
    push!(SubData.x, input)
    push!(SubData.z, cutData_dic[s].objVal)
    SubData.actItr = benders_obj.itr.cnt.i
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

function L1NormDict(x1::Dict)
    return sum(values(x1))
end




