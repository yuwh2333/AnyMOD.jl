function resDatatoDict!(resData_obj::resData)
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