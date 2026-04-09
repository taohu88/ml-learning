function engine = set_fields(engine, varargin)

args = varargin;
nargs = length(args);
for i=1:2:nargs
  switch args{i}
   case 'maximize', engine.maximize = args{i+1};
  end
end
