function val = get_field(engine, name)


switch name
 case 'bnet',      val = engine.bnet;
otherwise
  error(['invalid argument name ' name]);
end                                  
