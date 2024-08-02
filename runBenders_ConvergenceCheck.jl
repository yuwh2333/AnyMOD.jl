using Gurobi, AnyMOD, CSV, YAML, SlurmClusterManager, InteractiveUtils


include("./functions.jl")




# read-in of input options
par_df = CSV.read("settings_surro.csv",DataFrame)

if isempty(ARGS)
    id_int = 1
    t_int = 7
else
    id_int = parse(Int,ARGS[1])
    t_int = parse(Int,ARGS[2]) # number of threads
end

surroSelect_sym = Symbol(par_df[id_int,:surroSelect]) # can be :IDW
scr_int = par_df[id_int,:scr] # number of scenarios
res_int = par_df[id_int,:h] # number of hours
gap = 0.001


b = "C:/Users/23836/Desktop/git/EuSysMod/"

#region # * options for algorithm

# ! options for general algorithm

# target gap, number of iteration after unused cut is deleted, valid inequalities, number of iterations report is written, time-limit for algorithm, distributed computing?, surrogateBenders?, number of threads, optimizer
algSetup_obj = algSetup(gap, 20, (bal = false, st = false), 10, 120.0, true, true, t_int, Gurobi.Optimizer)

# ! options for stabilization

methKey_str = "qtr_5"

# write tuple for stabilization
stabMap_dic = YAML.load_file(b * "stabMap.yaml")
if methKey_str in keys(stabMap_dic)
	meth_tup = tuple(map(x -> Symbol(x[1]) => (; (Symbol(k) => v for (k, v) in x[2])...), collect(stabMap_dic[methKey_str]))...)
else
	meth_tup = tuple()
end

iniStab_ntup = (setup = :reduced, det = true) # options to initialize stabilization, :none for first input will skip stabilization, other values control input folders, second input determines, if heuristic model is solved stochastically or not

stabSetup_obj = stabSetup(meth_tup, 0.0, iniStab_ntup)
#stabSetup_obj = stabSetup(tuple(), 0.0, iniStab_ntup)


# ! options for near optimal

# defines objectives for near-optimal (can only take top-problem variables, must specify a variable)
nearOptSetup_obj = nothing # cost threshold to keep solution, lls threshold to keep solution, epsilon for near-optimal, cut deletion

#endregion

#region # * options for problem

# ! general problem settings
name_str =string("scr",scr_int, "_", res_int,"h_", surroSelect_sym, "_p", par_df[id_int,:p], "_gap",gap)
# name, temporal resolution, level of foresight, superordinate dispatch level, length of steps between investment years
info_ntup = (name = name_str, frs = 0, supTsLvl = 1, shortExp = 10) 

# ! input folders
dir_str = b




inDir_arr = [dir_str * "_basis",dir_str * "_full",dir_str * "timeSeries/" * string(res_int) * "hours_s" * string(scr_int), dir_str * "timeSeries/" * string(res_int) * "hours_det"] # input directory

if stabSetup_obj.ini.setup in (:none,:full) 
	heuInDir_arr = inDir_arr
elseif stabSetup_obj.ini.setup == :reduced
	heuInDir_arr =  [dir_str * "_basis",dir_str * "_heu",dir_str * "timeSeries/" * string(res_int) * "hours_s" * string(scr_int), dir_str * "timeSeries/" * string(res_int) * "hours_det"]
end 

inputFolder_ntup = (in = inDir_arr, heu = heuInDir_arr, results = dir_str * "results")

# ! scaling settings

scale_dic = Dict{Symbol,NamedTuple}()

scale_dic[:rng] = (mat = (1e-3,1e5), rhs = (1e-1,1e5))
scale_dic[:facHeu] = (capa = 1e2, capaStSize = 1e2, insCapa = 1e1, dispConv = 1e3, dispSt = 1e5, dispExc = 1e3, dispTrd = 1e3, costDisp = 1e1, costCapa = 1e2, obj = 1e0)
scale_dic[:facTop] = (capa = 1e2, capaStSize = 1e1, insCapa = 1e2, dispConv = 1e3, dispSt = 1e5, dispExc = 1e3, dispTrd = 1e3, costDisp = 1e1, costCapa = 1e0, obj = 1e3)
scale_dic[:facSub] = (capa = 1e0, capaStSize = 1e2, insCapa = 1e0, dispConv = 1e1, dispSt = 1e3, dispExc = 1e1, dispTrd = 1e1, costDisp = 1e0, costCapa = 1e2, obj = 1e1)

#endregion


#region # * prepare iteration
wrkCnt = scr_int #（equal to number of scenarios)

# initialize distributed computing
if algSetup_obj.dist 
    #addprocs(SlurmManager(; launch_timeout = 300), exeflags="--heap-size-hint=30G", nodes=1, ntasks=1, ntasks_per_node=1, cpus_per_task=4, mem_per_cpu="8G", time=4380) # add all available nodes
	#rmprocs(wrkCnt + 2) # remove one node again for main process
	addprocs(scr_int) 
    rmprocs(wrkCnt + 2)
	@suppress @everywhere begin 
		using AnyMOD, Gurobi
		runSubDist(w_int::Int64, s::Tuple{Int64,Int64}, resData_obj::resData, sol_sym::Symbol, optTol_fl::Float64=1e-8, crsOver_boo::Bool=false, wrtRes_boo::Bool=false) = Distributed.@spawnat w_int runSub(s, resData_obj, sol_sym, optTol_fl, crsOver_boo, wrtRes_boo)
	end
	passobj(1, workers(), [:info_ntup, :inputFolder_ntup, :scale_dic, :algSetup_obj])

else
	runSubDist = x -> nothing
end


# create benders object
benders_obj = bendersObj(info_ntup, inputFolder_ntup, scale_dic, algSetup_obj, stabSetup_obj, runSubDist, nearOptSetup_obj);

#endregion

#region # * iteration algorithm

# dataframe to track approximation of sub-problems
trackSub_df = DataFrame(i = Int[], Ts_dis = Int[], scr = Int[], actCost = Union{Nothing, Float64}[], estCost = Float64[], diff = Float64[], timeSub = Millisecond[], maxDiff = Bool[],sur = Float64[])
#dualvr = Dict{Tuple{Int64,Int64},Vector{Dict}}()
inputvr = Vector{Dict{Symbol, Float64}}()
push!(inputvr,Dict())
status = actStatus()
subData = Dict{Tuple{Int64,Int64},SubObj}()
for (id,s) in enumerate(collect(keys(benders_obj.sub))) subData[s] = SubObj() end
while true

	produceMessage(benders_obj.report.mod.options, benders_obj.report.mod.report, 1, " - Started iteration $(benders_obj.itr.cnt.i)", testErr = false, printErr = false)
    ############################################################
	#region # * solve top-problem

	str_time = now()
	resData_obj, stabVar_obj = @suppress runTop(benders_obj); 
	elpTop_time = now() - str_time


    
    #print bendersCuts
    if benders_obj.itr.cnt.i == 19
        printObject(benders_obj.top.parts.obj.cns[:bendersCuts],benders_obj.top)
    end
    

    #endregion#

    ##############################################################
    #region# decide which SP to solve

	# compare cutting plane estimator to surrogate esimator -> find largest difference
    # save the result of the top problem as input data for the subproblem	
	input = resDatatoDict(resData_obj)
    status.check_Conv = length(inputvr)>1 && L2NormDict(input,inputvr[length(inputvr)]) < sqrt(10*length(input)) ? true : false
    #save capacity variables
    push!(inputvr,input)
    
    sMaxDiff_tup, cutVar_df = computeSurrogates(benders_obj, surroSelect_sym, input, par_df, subData)
    
    cut_group = collect(keys(benders_obj.sub))
    if surroSelect_sym == :all_cc || benders_obj.itr.cnt.i == 2 || status.check_Conv == true || surroSelect_sym == :simu || benders_obj.itr.gap < gap || surroSelect_sym == :dist
        cut_group = collect(keys(benders_obj.sub))
    else 
        empty!(cut_group)
    #enforce a maximum distance after that a SP must be solved again
        #for (id,s) in enumerate(collect(keys(benders_obj.sub)))
        #    if L2NormDict(inputvr[subData[s].actItr],input) > 5 * L1NormDict(input)
        #        push!(cut_group, s)
        #    end
        #end
        #enforce a minimum distance before that a SP should not be solved again
        #if  (L2NormDict(inputvr[subData[sMaxDiff_tup].actItr],input) > 0.0001 * L1NormDict(input)) && !(sMaxDiff_tup in cut_group)  
        push!(cut_group, sMaxDiff_tup) 
        #end
    end
    #if isempty(cut_group)
    #    cut_group = collect(keys(benders_obj.sub))
    #end
    #record which scenario is in cut_group
    insertcols!(cutVar_df, :actCost => Vector{Union{Nothing, Float64}}(nothing, nrow(cutVar_df)))
    cutVar_df[!,:maxDiff] = map(x -> (x.Ts_dis, x.scr) in cut_group, eachrow(cutVar_df))

 
    #endregion
    
    ####################################################################################
    #region# * solve and get results from subproblems

	# start solving sub-problems
	cutData_dic = Dict{Tuple{Int64,Int64},resData}()
	timeSub_dic = Dict{Tuple{Int64,Int64},Millisecond}()
	lss_dic = Dict{Tuple{Int64,Int64},Float64}()

	if benders_obj.algOpt.dist futData_dic = Dict{Tuple{Int64,Int64},Future}() end
	for (id,s) in enumerate(collect(keys(benders_obj.sub)))
		if benders_obj.algOpt.dist # distributed case
			futData_dic[s] = runSubDist(id + 1, s, copy(resData_obj), :barrier, 1e-8)
            println("worker",id+1, "_executes job",s)
            
            #cutData_dic[s], timeSub_dic[s], lss_dic[s] = fetch(futData_dic[s])
		else # non-distributed case
			if s in cut_group || surroSelect_sym == :NN_simu || surroSelect_sym == :IDW_simu 
			    cutData_dic[s], timeSub_dic[s], lss_dic[s], = runSub(benders_obj.sub[s], s, copy(resData_obj), :barrier, 1e-8)
                cutVar_df[(cutVar_df[!,:Ts_dis].== s[1]) .& (cutVar_df[!,:scr] .== s[2]), :actCost] .= cutData_dic[s].objVal
                cutVar_df[(cutVar_df[!,:Ts_dis].== s[1]) .& (cutVar_df[!,:scr] .== s[2]), :timeSub] .= timeSub_dic[s]
                #save the input and output in Points for subproblems solved as previous data
                if s in cut_group
                    savePoint!(subData[s], input, cutData_dic, s, benders_obj)
                end
            else
                cutData_dic[s] = resData()
                cutData_dic[s].objVal = cutVar_df[(cutVar_df[!,:Ts_dis].== s[1]) .& (cutVar_df[!,:scr] .== s[2]), :sur][1]
			end	
		end		
	end
	
    
	# top-problem without stabilization
	if !isnothing(benders_obj.stab) && benders_obj.nearOpt.cnt == 0 @suppress runTopWithoutStab!(benders_obj) end
    

	# get results of sub-problems
	
    if benders_obj.algOpt.dist
		wait.(collect(values(futData_dic)))
		for s in collect(keys(benders_obj.sub))
			cutData_dic[s], timeSub_dic[s], lss_dic[s], = fetch(futData_dic[s])
            savePoint!(subData[s], input, cutData_dic, s, benders_obj)
		end
	end
    

    #endregion#

    #########################################################
    #region # * analyse results and update refinements
    
    # ensure the stabilization center can be correctly updated
    if cut_group == collect(keys(benders_obj.sub)) && benders_obj.itr.cnt.i>3
        benders_obj.stab.objVal = status.last_stab_obj
        benders_obj.itr.best.objVal = status.real_benders_best_obj
    end

    #for simulation, use surrogates to update the stabilization center
    if surroSelect_sym == :NN_simu || surroSelect_sym == :IDW_simu
        for (id,s) in enumerate(collect(keys(benders_obj.sub)))
            if !(s in cut_group)
                cutData_dic[s].objVal = cutVar_df[(cutVar_df[!,:Ts_dis].== s[1]) .& (cutVar_df[!,:scr] .== s[2]), :sur][1]
            end
        end
    end
    
    
    # update results and stabilization
	updateIteration!(benders_obj, cutData_dic, stabVar_obj)

    #Use real information for convergence check
    if cut_group == collect(keys(benders_obj.sub))
        status.real_benders_best_obj = benders_obj.itr.best.objVal
        status.rtn_boo = checkConvergence(benders_obj, lss_dic)
    else
        benders_obj.itr.res[:curBest] = status.real_benders_best_obj
        benders_obj.itr.gap = benders_obj.nearOpt.cnt == 0 ? (1 - benders_obj.itr.res[:lowLimCost] / benders_obj.itr.res[:actTotCost]) : abs(benders_obj.itr.res[:actTotBest] / benders_obj.itr.res[:optCost])	
    end

    if surroSelect_sym == :NN_simu || surroSelect_sym == :IDW_simu
        real_actTotCost = sum(cutVar_df[!,:actCost]) + benders_obj.itr.res[:topCost]
        benders_obj.itr.gap = 1 - benders_obj.itr.res[:lowLimCost] / real_actTotCost
    end
        
	# report on iteration
	reportBenders!(benders_obj, resData_obj, elpTop_time, timeSub_dic, lss_dic)
    
    if surroSelect_sym == :NN_simu || surroSelect_sym == :IDW_simu
        benders_obj.itr.gap = benders_obj.nearOpt.cnt == 0 ? (1 - benders_obj.itr.res[:lowLimCost] / benders_obj.itr.res[:actTotCost]) : abs(benders_obj.itr.res[:actTotBest] / benders_obj.itr.res[:optCost])	
    end

	#endregion

    #################################################
    #add actual costs for subproblems really solved
    
    for row in eachrow(cutVar_df)
        if (row[:Ts_dis], row[:scr]) in cut_group 
            row[:actCost] = cutData_dic[(row[:Ts_dis], row[:scr])].objVal
            row[:timeSub] = timeSub_dic[(row[:Ts_dis], row[:scr])]     
        end
    end   
    
   
    #simulation
    if surroSelect_sym == :simu
        cutVar_df[!,:diff] = cutVar_df[!,:actCost]  .- cutVar_df[!,:estCost]
        sMaxDiff_tup = tuple((cutVar_df[findall(maximum(cutVar_df[!,:diff]) .== cutVar_df[!,:diff]), :] |> (z -> map(x -> z[1,x], [:Ts_dis, :scr])))...)
    end
    
    #define specific cuts
    if benders_obj.itr.cnt.i>2 && benders_obj.algOpt.dist == false
        if surroSelect_sym == :simu 
            filter!(x -> x[1] == sMaxDiff_tup, benders_obj.cuts)
        else
            filter!(x -> x[1] in cut_group, benders_obj.cuts)
        end
    end

	# add number of iteration and add to overall dataframe
	cutVar_df[!,:i] .= benders_obj.itr.cnt.i
	append!(trackSub_df, cutVar_df)

  
	if status.rtn_boo break end
	benders_obj.itr.cnt.i = benders_obj.itr.cnt.i + 1
    #report trackSub_df
    # Define a transformation function to replace `nothing` with a placeholder
    transform_nothing = (col, val) -> val === nothing ? "NA" : val
    if benders_obj.itr.cnt.i % 10 == 0 
        CSV.write(benders_obj.report.mod.options.outDir * "/trackingSub_$(benders_obj.info.name).csv", trackSub_df;transform=(col, val) -> transform_nothing(col, val))
    end
    if benders_obj.itr.cnt.i == 300 break end
end

#endregion

# print results

benders_obj.report.itr[!,:run] .= benders_obj.info.name
trackSub_df[!,:run] .= benders_obj.info.name

#print capacities
capa_track  = DataFrame(i = Int64[], Symbol = Symbol[], value = Float64[])
for (i,element) in enumerate(inputvr)
    for (key, value) in pairs(inputvr[i])
        append!(capa_track, DataFrame(i = i, Symbol = key, value = value))
    end
end

transform_nothing = (col, val) -> val === nothing ? "NA" : val

CSV.write(benders_obj.report.mod.options.outDir * "/iterationBenders_$(benders_obj.info.name).csv", benders_obj.report.itr)
CSV.write(benders_obj.report.mod.options.outDir * "/trackingSub_$(benders_obj.info.name).csv", trackSub_df; transform=(col, val) -> transform_nothing(col, val))
CSV.write(benders_obj.report.mod.options.outDir * "/trackingCapa_$(benders_obj.info.name).csv", capa_track)
