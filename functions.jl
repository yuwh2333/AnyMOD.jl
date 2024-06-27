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
    function actStatus()
        new(false, false, Inf, Inf)
    end
end

mutable struct SubObj
    x:: Vector{Dict}
    z:: Vector{Float64}
    actItr:: Int64
    function SubObj()
        sub_obj = new()
        sub_obj.x = Vector{Dict}()
        sub_obj.z = Vector{Float64}()
        sub_obj.actItr = 0
        return sub_obj
    end
end

function savePoint!(SubData, input, cutData_dic,s)
    push!(SubData.x, input)
    push!(SubData.z, cutData_dic[s].objVal)
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




