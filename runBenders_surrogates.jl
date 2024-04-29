using AnyMOD, Gurobi, CSV, YAML
include("./IDW.jl")



# read-in of input options
par_df = CSV.read("settings_surro.csv",DataFrame)

if isempty(ARGS)
    id_int = 1
    t_int = 4
else
    id_int = parse(Int,ARGS[1])
    t_int = parse(Int,ARGS[2]) # number of threads
end

surroSelect_sym = Symbol(par_df[id_int,:surroSelect]) # can be :n-1, :IDW
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
name_str =string("scr",scr_int, "_", res_int,"h_", cutSelect_sym,"_gap",gap)
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
trackSub_df = DataFrame(i = Int[], Ts_dis = Int[], scr = Int[], actCost = Float64[], estCost = Float64[], diff = Float64[], timeSub = Millisecond[], maxDiff = Bool[],sur = Float64[], solved = Bool[])
sMaxDiff_tup = tuple()
Points_x = Dict{Tuple{Int64,Int64},Vector{Dict}}() #Input of each subproblem of previous iterations are documented
Points_y = Dict{Tuple{Int64,Int64}, Vector{Float64}}() #output of each subproblem of previous iterations
cut_group = collect(keys(benders_obj.sub)) 
trackSub_itr = Vector{DataFrame}()
while true

	produceMessage(benders_obj.report.mod.options, benders_obj.report.mod.report, 1, " - Started iteration $(benders_obj.itr.cnt.i)", testErr = false, printErr = false)

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


	#endregion

	#region # * analyse results and update refinements

	# update results and stabilization
	updateIteration!(benders_obj, cutData_dic, stabVar_obj)

	# report on iteration
	reportBenders!(benders_obj, resData_obj, elpTop_time, timeSub_dic, lss_dic)

	# check convergence and finish
	rtn_boo = checkConvergence(benders_obj, lss_dic)
	
	#endregion

     #save the input data of the subproblem	
	input = Dict{Symbol, Float64}() 
    for sys in (:tech, :exc)
		for sSym in keys(resData_obj.capa[sys]), capaSym in keys(resData_obj.capa[sys][sSym])
            var_name = Symbol(string(sys), string(sSym), string(capaSym))
            input[var_name] = resData_obj.capa[sys][sSym][capaSym].value[1]                    
		end		
	end
    #save in Points as previous 
    for (id,s) in enumerate(collect(keys(benders_obj.sub))) 
        #point_df[s]= DataFrame(key = keys(input), value = values(input))
        if s in cut_group
            #point_df[!,:objValue] .= cutData_dic[s].objVal
            if haskey(Points_x, s)
                push!(Points_x[s], input)
            else
                Points_x[s] = [input]
            end
            if haskey(Points_y, s)
                push!(Points_y[s], cutData_dic[s].objVal)
            else
                Points_y[s] = [cutData_dic[s].objVal]
            end
        end
    end

	# add the actual costs from sub-problems (should be after sub-problems are solved in distributed case!)
	cutVar_df[!,:actCost] = map(x -> cutData_dic[(x.Ts_dis, x.scr)].objVal, eachrow(cutVar_df))
	cutVar_df[!,:timeSub] = map(x -> timeSub_dic[(x.Ts_dis, x.scr)], eachrow(cutVar_df))
    cutVar_df[!,:sur] .= 0.0
    for row in eachrow(cutVar_df)
        if ((row.Ts_dis,row.scr) in cut_group)
            row.sur = row.actCost
        else 
            if surroSelect_sym == :IDW
                row.sur = IDW.computeIDW(Points_x[(row.Ts_dis, row.scr)], Points_y[(row.Ts_dis, row.scr)], input)
            end
            if surroSelect_sym == :n-1
                temp_df = track_itr[length(trackSub_itr)]
                row.sur = temp_df[(temp_df.Ts_dis .== row.Ts_dis) .& (temp_df.scr .== row.scr), :sur]
            end
        end
    end
    cutVar_df[!,:diff] = cutVar_df[!,:sur]  .- cutVar_df[!,:estCost]

	# find case with biggest difference
	sMaxDiff_tup = tuple((cutVar_df[findall(maximum(cutVar_df[!,:diff]) .== cutVar_df[!,:diff]), :] |> (z -> map(x -> z[1,x], [:Ts_dis, :scr])))...)
	cutVar_df[!,:maxDiff] = map(x -> sMaxDiff_tup == (x.Ts_dis, x.scr), eachrow(cutVar_df))
    cutVar_df[!,:solved] = map(x -> (x.Ts_dis,x.scr) in cut_group, eachrow(cutVar_df))
    #define cut group
    empty!(cut_group)
    for row in eachrow(cutVar_df)
        if row.diff<-1 push!(cut_group, (row.Ts_dis,row.scr)) end
    end
    push!(cut_group, sMaxDiff_tup)
    

	# add number of iteration and add to overall dataframe
	cutVar_df[!,:i] .= benders_obj.itr.cnt.i
	append!(trackSub_df, cutVar_df)
    push!(trackSub_itr, cutVar_df)

    #define specific cuts
    filter!(x -> x[1] in cut_group, benders_obj.cuts)


	if rtn_boo break end
	benders_obj.itr.cnt.i = benders_obj.itr.cnt.i + 1

end

#endregion

# print results

benders_obj.report.itr[!,:run] .= benders_obj.info.name
trackSub_df[!,:run] .= benders_obj.info.name

CSV.write(benders_obj.report.mod.options.outDir * "/iterationBenders_$(benders_obj.info.name).csv", benders_obj.report.itr)
CSV.write(benders_obj.report.mod.options.outDir * "/trackingSub_$(benders_obj.info.name).csv", trackSub_df)
CSV.write(benders_obj.report.mod.options.outDir * "/trackingCapa_$(benders_obj.info.name).csv", trackCapa_df)