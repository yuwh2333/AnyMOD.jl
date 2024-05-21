using AnyMOD, Gurobi, CSV, YAML
include("./IDW.jl")
include("./functions.jl")




# read-in of input options
par_df = CSV.read("settings_surro.csv",DataFrame)

if isempty(ARGS)
    id_int = 1
    t_int = 4
else
    id_int = parse(Int,ARGS[1])
    t_int = parse(Int,ARGS[2]) # number of threads
end

surroSelect_sym = Symbol(par_df[id_int,:surroSelect]) # can be :IDW
scr_int = par_df[id_int,:scr] # number of scenarios
res_int = par_df[id_int,:h] # number of hours
gap = 0.01


b = "C:/Users/23836/Desktop/git/EuSysMod/"

#region # * options for algorithm

# ! options for general algorithm

# target gap, number of iteration after unused cut is deleted, valid inequalities, number of iterations report is written, time-limit for algorithm, distributed computing?, number of threads, optimizer
algSetup_obj = algSetup(gap, 20, (bal = false, st = false), 10, 120.0, false, t_int, Gurobi.Optimizer)

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


# ! options for near optimal

# defines objectives for near-optimal (can only take top-problem variables, must specify a variable)
nearOptSetup_obj = nothing # cost threshold to keep solution, lls threshold to keep solution, epsilon for near-optimal, cut deletion

#endregion

#region # * options for problem

# ! general problem settings
name_str =string("scr",scr_int, "_", res_int,"h_", surroSelect_sym,"_gap",gap)
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

# initialize distributed computing
if algSetup_obj.dist 
	addprocs(scr_int*2) 
	@suppress @everywhere begin 
		using AnyMOD, Gurobi
		runSubDist(w_int::Int64, resData_obj::resData, sol_sym::Symbol, optTol_fl::Float64=1e-8, crsOver_boo::Bool=false, wrtRes_boo::Bool=false) = Distributed.@spawnat w_int runSub(sub_m, resData_obj, sol_sym, optTol_fl, crsOver_boo, wrtRes_boo)
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
trackSub_df = DataFrame(i = Int[], Ts_dis = Int[], scr = Int[], actCost = Float64[], estCost = Float64[], diff = Float64[], timeSub = Millisecond[], maxDiff = Bool[],sur = Float64[])
sMaxDiff_tup = tuple()
Points = PointsData() #documentation of input and output of previous iterations
cut_group = collect(keys(benders_obj.sub)) 
dualvr = Dict{Tuple{Int64,Int64},Vector{Dict}}()
inputvr = Vector{Dict{Symbol, Float64}}()
track_itr = Vector{DataFrame}()
while true

	produceMessage(benders_obj.report.mod.options, benders_obj.report.mod.report, 1, " - Started iteration $(benders_obj.itr.cnt.i)", testErr = false, printErr = false)

    ##############################################################
	#region # * solve top-problem and (start) sub-problems

	str_time = now()
	resData_obj, stabVar_obj = @suppress runTop(benders_obj); 
	elpTop_time = now() - str_time

	# start solving sub-problems
	cutData_dic = Dict{Tuple{Int64,Int64},resData}()
	timeSub_dic = Dict{Tuple{Int64,Int64},Millisecond}()
	lss_dic = Dict{Tuple{Int64,Int64},Float64}()

	if benders_obj.algOpt.dist futData_dic = Dict{Tuple{Int64,Int64},Future}() end
	for (id,s) in enumerate(collect(keys(benders_obj.sub)))
		if benders_obj.algOpt.dist # distributed case
			futData_dic[s] = runSubDist(id + 1, copy(resData_obj), :barrier, 1e-8)
		else # non-distributed case
			cutData_dic[s], timeSub_dic[s], lss_dic[s] = runSub(benders_obj.sub[s], copy(resData_obj), :barrier, 1e-8)
		end		
	end
	
	# get the estimated cost from top-problem (must be before running top problem again without stabilization!)
	cutVar_df = copy(benders_obj.top.parts.obj.var[:cut]) 
	cutVar_df[!,:estCost] = value.(cutVar_df)[!,:var]
	select!(cutVar_df,Not([:var]))

	# top-problem without stabilization
	if !isnothing(benders_obj.stab) && benders_obj.nearOpt.cnt == 0 @suppress runTopWithoutStab!(benders_obj) end

	# get results of sub-problems
	if benders_obj.algOpt.dist
		wait.(collect(values(futData_dic)))
		for s in collect(keys(benders_obj.sub))
			cutData_dic[s], timeSub_dic[s], lss_dic[s] = fetch(futData_dic[s])
		end
	end

    #save the result of top-problem as input data for the subproblem	
	input = resDatatoDict(resData_obj) #convert from type resData to Dict
    push!(inputvr,input)

    #save dual, not currently relevant
    #saveDual(cutData_dic, cut_group, resData_obj, benders_obj, dualvr)

    #endregion
   
    #################################################################
	#region # * analyse results and update refinements

	# update results and stabilization
	updateIteration!(benders_obj, cutData_dic, stabVar_obj)

	# report on iteration
	reportBenders!(benders_obj, resData_obj, elpTop_time, timeSub_dic, lss_dic)

	# check convergence and finish
	rtn_boo = checkConvergence(benders_obj, lss_dic)
	
	#endregion

    #####################################################
    #region# * compute track_df / surrogates / difference

	# add the actual costs from sub-problems (should be after sub-problems are solved in distributed case!)
	cutVar_df[!,:actCost] = map(x -> cutData_dic[(x.Ts_dis, x.scr)].objVal, eachrow(cutVar_df))
	cutVar_df[!,:timeSub] = map(x -> timeSub_dic[(x.Ts_dis, x.scr)], eachrow(cutVar_df))
    cutVar_df[!,:sur] .= 0.0
    if benders_obj.itr.cnt.i>2
        #compute surrogates
        for row in eachrow(cutVar_df)
            if surroSelect_sym == :IDW
                row.sur = computeIDW(Points.x[(row.Ts_dis, row.scr)], Points.y[(row.Ts_dis, row.scr)], input)
            end
            if surroSelect_sym == :NN
                row.sur = computeNN(Points.x[(row.Ts_dis, row.scr)], Points.y[(row.Ts_dis, row.scr)], input)
            end
            if surroSelect_sym == :dualIDW
                row.sur = computedualIDW(Points.x[(row.Ts_dis, row.scr)], Points.y[(row.Ts_dis, row.scr)], input, dualvr[(row.Ts_dis, row.scr)])
            end
        end
    else
        cutVar_df[!,:sur] .= 0
    end
    cutVar_df[!,:diff] = cutVar_df[!,:sur]  .- cutVar_df[!,:estCost]
    
    #handel the situation the difference is negative
    #cutVar_df[!,:diff] = map(x -> abs(x.diff), eachrow(cutVar_df))
    for row in eachrow(cutVar_df)
        if row.diff<-0.1 
            row.diff = 1.0
        end
    end

    #endregion

    ########################################################################
    #region # * define sMaxDiff_tup / cut_group

	# find case with biggest difference
	sMaxDiff_tup = tuple((cutVar_df[findall(maximum(cutVar_df[!,:diff]) .== cutVar_df[!,:diff]), :] |> (z -> map(x -> z[1,x], [:Ts_dis, :scr])))...)
	cutVar_df[!,:maxDiff] = map(x -> sMaxDiff_tup == (x.Ts_dis, x.scr), eachrow(cutVar_df))
    if benders_obj.itr.cnt.i>2
        empty!(cut_group)
        push!(cut_group, sMaxDiff_tup)
    end
    
    #endregion

    ########################################################################################

    #save the input and output of subproblems solved as previous data for later prediction
    updatePoints!(Points, input, cutData_dic, benders_obj)

    #delete specific cuts
    if benders_obj.itr.cnt.i>2 
        filter!(x -> x[1] in cut_group, benders_obj.cuts)
    end

	# add number of iteration and add to overall dataframe
	cutVar_df[!,:i] .= benders_obj.itr.cnt.i
	append!(trackSub_df, cutVar_df)
    push!(track_itr, cutVar_df)


	if rtn_boo break end

    #update iteration numbers
	benders_obj.itr.cnt.i = benders_obj.itr.cnt.i + 1

    #report trackSub_df
    if benders_obj.itr.cnt.i % 10 == 0 
        CSV.write(benders_obj.report.mod.options.outDir * "/trackingSub_$(benders_obj.info.name).csv", trackSub_df)
    end
    #stop iteration
    if benders_obj.itr.cnt.i == 200 break end
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

CSV.write(benders_obj.report.mod.options.outDir * "/iterationBenders_$(benders_obj.info.name).csv", benders_obj.report.itr)
CSV.write(benders_obj.report.mod.options.outDir * "/trackingSub_$(benders_obj.info.name).csv", trackSub_df)
CSV.write(benders_obj.report.mod.options.outDir * "/trackingCapa_$(benders_obj.info.name).csv", capa_track)