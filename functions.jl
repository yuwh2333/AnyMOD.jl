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

mutable struct PointsData
    x::Dict{Tuple{Int64,Int64}, Vector{Dict}}
    y::Dict{Tuple{Int64,Int64}, Vector{Float64}}
    function PointsData()
        new(Dict{Tuple{Int64,Int64}, Vector{Dict}}(), Dict{Tuple{Int64,Int64}, Vector{Float64}}())
    end
end

function savePoint!(Points, input, cutData_dic,s)
    if haskey(Points.x, s)
        push!(Points.x[s], input)
    else
        Points.x[s] = [input]
    end
    if haskey(Points.y, s)
        push!(Points.y[s], cutData_dic[s].objVal)
    else
        Points.y[s] = [cutData_dic[s].objVal]
    end
end




