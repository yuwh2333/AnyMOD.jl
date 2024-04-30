using AnyMOD, Gurobi, CSV, YAML

b = "C:/Users/23836/Desktop/Git/EuSysMod/"

#region # * options for algorithm

# ! options for general algorithm

# target gap, number of iteration after unused cut is deleted, valid inequalities, number of iterations report is written, time-limit for algorithm, distributed computing?, number of threads, optimizer
algSetup_obj = algSetup(0.01, 20, (bal = false, st = false), 10, 120.0, false, 4, Gurobi.Optimizer)

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
name_str ="_test"
# name, temporal resolution, level of foresight, superordinate dispatch level, length of steps between investment years
info_ntup = (name = name_str, frs = 0, supTsLvl = 1, shortExp = 10) 

# ! input folders
dir_str = b
scr_int = 2 # number of scenarios
res_int = 96

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

#variables for printing track
track_df = DataFrame(i = Int[], id = Int[], s = Tuple[], Ts_dis = Int[], scr = Int[], actCost = Float64[], estCost = Float64[], diff = Float64[], timeSub = Millisecond[], largest = Float64[])
track_itr = Vector{DataFrame}() 
sLargeSetup = true
single_cut = false
sLargeDif_tup = Tuple[]
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

	track_itr_temp = DataFrame(i = Int[], id = Int[], s = Tuple[], Ts_dis = Int[], scr = Int[], actCost = Float64[], estCost = Float64[], diff = Float64[], timeSub = Millisecond[], largest = Float64[])

	if benders_obj.algOpt.dist futData_dic = Dict{Tuple{Int64,Int64},Future}() end
	if sLargeSetup == true && single_cut == true && !isempty(sLargeDif_tup)
		s = sLargeDif_tup
		cutData_dic[s], timeSub_dic[s], lss_dic[s] = runSub(benders_obj.sub[s], copy(resData_obj), :barrier, 1e-8)
	else 
		for (id,s) in enumerate(collect(keys(benders_obj.sub)))
			if benders_obj.algOpt.dist # distributed case
				futData_dic[s] = runSubDist(id + 1, copy(resData_obj), :barrier, 1e-8)
			else # non-distributed case
				cutData_dic[s], timeSub_dic[s], lss_dic[s] = runSub(benders_obj.sub[s], copy(resData_obj), :barrier, 1e-8)
				#print the traking data in this iteration
				cutVar_df = copy(benders_obj.top.parts.obj.var[:cut]) #get the data of estimated cost
				cutVar_df[!,:estCosts] = value.(cutVar_df)[!,:var]
				actCost_temp = cutData_dic[s].objVal
				estCost_temp = cutVar_df[(cutVar_df.Ts_dis .== s[1]) .& (cutVar_df.scr .==s[2]), :estCosts][1]
				new_row = (i= benders_obj.itr.cnt.i, id = id, s = s, Ts_dis = s[1], scr = s[2], actCost = actCost_temp, estCost = estCost_temp, diff = actCost_temp - estCost_temp, timeSub = timeSub_dic[s], largest = NaN)
				push!(track_itr_temp, new_row)
		    end
		end		
	end
	
	#determine the scenario with largest gap between actual cost and estimated cost
	if sLargeSetup == false || single_cut == false
		for i in 1:nrow(track_itr_temp)
			if i == argmax(track_itr_temp[!,:diff]) 
				track_itr_temp[i,:largest] = 1	
				sLargeDif_tup = (track_itr_temp[i, :s])
				single_cut = true
			else
				track_itr_temp[i,:largest] = 0
			end
		end
		push!(track_itr, track_itr_temp)
	else
		sLargeDif_tup = ()
		single_cut = false
	end
	println(single_cut)
	
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

	#if sLargeSetup == true && single_cut == true
	#	filter!(x -> x[1] == sLargeDif_tup, benders_obj.cuts)
	#	single_cut = false
	#end

	# report on iteration
	reportBenders!(benders_obj, resData_obj, elpTop_time, timeSub_dic, lss_dic)

	# check convergence and finish
	rtn_boo = checkConvergence(benders_obj, lss_dic)
	
	#endregion

	if rtn_boo break end
	benders_obj.itr.cnt.i = benders_obj.itr.cnt.i + 1
	

end	#endofiteration

#print tracking results
for itr in track_itr
	append!(track_df, itr)
end
CSV.write("track_df.csv",track_df)


benders_obj.report.itr
#endregion



#filter!(x -> x[1] == sLargeDif_tup, benders_obj.cuts)

