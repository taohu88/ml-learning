function p = plotdigraph( G, varargin )
%PLOTGRAPH custom plot graph
    nargin = length(varargin);
    if nargin > 0 && mod(nargin,2) ~= 0
        error('parameters must have values.');
    end
    nodefont = 10;
    markersz = 10;
    plateedges ={};
    if nargin > 0
        for i=1:2:nargin
            switch varargin{i}
                case 'MarkerSize'
                    markersz = varargin{i+1};
                case 'NodeFont'
                    nodefont = varargin{i+1};
                case 'PlateEdges'
                    plateedges = varargin{i+1};
            end
                    
        end
    end

    figure;
    p = plot(G);
    p.MarkerSize = markersz;
    names = p.NodeLabel;
    p.NodeLabel = {};
    datalen = length(p.XData);
    for i=1:datalen
        text(p.XData(i)+markersz/200,p.YData(i),names(i),'FontSize',nodefont);       
    end
    layout(p,'layered');
    hold on;
    for i=1:size(plateedges,1)
        v1 = [p.XData(plateedges{i,1}(1)),p.YData(plateedges{i,1}(1))]';
        v2 = [p.XData(plateedges{i,1}(2)),p.YData(plateedges{i,1}(2))]';
        a = atan2d(v1(2)-v2(2),v1(1)-v2(1));
        r1 = 0.35;
        r2 = 0.25;
        r3 = 0.2;
        box = [v1(1)+r1*cosd(a)-r1*sind(a),v1(2)+r1*sind(a)+r1*cosd(a);
               v1(1)+r1*cosd(a)+r1*sind(a),v1(2)+r1*sind(a)-r1*cosd(a);
               v2(1)-r1*cosd(a)+r1*sind(a),v2(2)-r1*sind(a)-r1*cosd(a);
               v2(1)-r1*cosd(a)-r1*sind(a),v2(2)-r1*sind(a)+r1*cosd(a);
               v1(1)+r1*cosd(a)-r1*sind(a),v1(2)+r1*sind(a)+r1*cosd(a)]';        
        plot(box(1,:),box(2,:),'k-');
        h=text(v2(1)-r3*cosd(a)+r3*sind(a),v2(2)-r2*sind(a)-r2*cosd(a),plateedges{i,2},'FontSize',12);
        set(h,'Rotation',a-90);
    end
    hold off;
end

