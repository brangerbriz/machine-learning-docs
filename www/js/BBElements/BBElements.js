/*
      +++++++++++++
    +++++++++++++++++
  +++++++++++++++++++++
 +++++++ ---------- ++++       ____                                         ____       _
++++++++|  ______  |+++++     |  _ \                                       |  _ \     (_)
++++++__| |______| |+++++     | |_) |_ __ __ _ _ __   __ _  ___ _ __       | |_) |_ __ _ ____
+++++|  _________  |+++++     |  _ <| '__/ _` | '_ \ / _` |/ _ \ '__|      |  _ <| '__| |_  /
+++++| |_________| |+++++     | |_) | | | (_| | | | | (_| |  __/ |         | |_) | |  | |/ /
 ++++|_____________|++++      |____/|_|  \__,_|_| |_|\__, |\___|_| _______ |____/|_|  |_/___|
  +++++++++++++++++++++                              __ | |       |_______|
    +++++++++++++++++                                \___/
      +++++++++++++
*/
function BBElements(){

    /* * * * * * * * * * * * * * * * * UTILS * * * * * * * * * * * * * * * *  */
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */

    function isUsingCSSFile(filename){
        let isUsing = false
        let cssFiles = document.querySelectorAll('link')
        for (let i = 0; i < cssFiles.length; i++) {
            if(cssFiles[i].getAttribute('href').indexOf(filename) >= 0){
                isUsing = true
                break
            }
        }
        return isUsing
    }

    function mkLink(txt,url){
        let a = document.createElement('a')
            a.textContent = txt
            a.setAttribute('href',url)
            a.setAttribute('target','_blank')
        return a
    }

    function mkSpan(txt){
        let span = document.createElement('span')
            span.textContent = txt
        return span
    }

    function md2html( str, parent ){
        let links = []
        let alt = str.replace(/\[([^\]]*)\]\(([^)]*)\)/g,(s,txt,lnk)=>{
            let i = str.indexOf(s)
            let o = i + s.length
            links.push({ in:i, out:o, str:s, txt:txt, url:lnk})
            return '_|_'
        })
        let strings = alt.split('_|_')
        if(links.length>0){
            links.forEach((l,i)=>{
                if(l.in!=0) parent.appendChild( mkSpan(strings[i]) )
                parent.appendChild( mkLink(l.txt,l.url) )
            })
            parent.appendChild( mkSpan(strings[strings.length-1]) )
        } else {
            parent.textContent = str
        }
    }

    /*
        HANDLE MEDIA CONTAINER ELEMENTS
        -------------------------------
        find all <section class="media">
        if they have child <img> w/an alt attribute,
        then creates caption divs from alt content
        alt values with [markdown](links) will be converted to <a> tags
    */

    // add show class to images after loading (if using animations)
    if( isUsingCSSFile('bb-animations.css') ){
        let imgs = document.querySelectorAll('img')
        for (let i = 0; i < imgs.length; i++) {
            if( imgs[i].parentNode.className == "media" ){
                if(imgs[i].complete){
                    imgs[i].classList.add('show')
                } else {
                    imgs[i].addEventListener('load',function(){
                        this.classList.add('show')
                    })
                }
            }
        }
    }

    let medias = document.querySelectorAll('section.media')
    for (let i = 0; i < medias.length; i++) {
        let m = medias[i]
        // create img caption elements
        let img = m.querySelector('img')
        let cap = m.querySelector('div.caption')
        if(!cap && img && img.getAttribute('alt')){
            cap = document.createElement('div')
            cap.className = 'caption'
            if(m.dataset.fullwidth) cap.setAttribute('data-fullwidth','true')
            let txt = img.getAttribute('alt')
            md2html(txt,cap)
            m.appendChild(cap)
        }
    }

    /*
        HANDLE QUOTE CONTAINER ELEMENTS
        -------------------------------
        find all <span class="quote">
        if they have a data-credit attribute,
        then creates credit divs from the value
        data-credit with [markdown](links) will be converted to <a> tags
    */
    let quotes = document.querySelectorAll('span.quote')
    for (let i = 0; i < quotes.length; i++) {
        let q = quotes[i]
        // create quote credit elements
        let val = q.getAttribute('data-credit')
        let cred = q.querySelector('div.quote-credit')
        if(!cred && val){
            cred = document.createElement('div')
            cred.className = 'quote-credit'
            md2html(val,cred)
            q.appendChild(cred)
        }
    }

    /*
        HANDLE MARGINAL NOTES
        -------------------------------
        finds all <span class="marginal-note"> and numbers them
        then it creates the <aside> per <p> containing span.marginal-note
        and adds the info in it's data-info property to that <aside>
    */
    function makeNote( num, info ) {
        let s1 = document.createElement('span')
        let s2 = document.createElement('span')
        s1.style.color = "#e40477"
        s1.textContent = num+'. '
        md2html(info,s2)
        let span = document.createElement('span')
        span.style.marginBottom = "10px"
        span.style.display = "block"
        span.appendChild( s1 )
        span.appendChild( s2 )
        return span
    }

    let notes = document.querySelectorAll('span.marginal-note')
    for (let i = 0; i < notes.length; i++) {
        // add note numbers in <p> tags
        notes[i].textContent = i+1
        // target aside, or create aside
        let p = notes[i].parentNode
        let aTop = p.querySelector('aside.note')
        let aBottom = p.querySelector('aside.note.bottom')
        if(!aTop){
            aTop = document.createElement('aside')
            aTop.className = 'note'
            p.insertBefore(aTop, p.childNodes[0])
            aBottom = document.createElement('aside')
            aBottom.className = 'note bottom'
            p.appendChild(aBottom)
        } else if(aTop.firstChild || aBottom.firstChild){
            // if they already have notes in them, remove them
            if(aTop.firstChild){
                while (aTop.firstChild) {
                    aTop.removeChild(aTop.firstChild)
                }
            } else if(aBottom.firstChild){
                while (aBottom.firstChild) {
                    aBottom.removeChild(aBottom.firstChild)
                }
            }
        }
        // append individual note
        let nfo = notes[i].getAttribute('data-info')
        aTop.appendChild( makeNote(i+1,nfo) )
        aBottom.appendChild( makeNote(i+1,nfo) )
    }

    /*
        syntax highlight with highlightJS
        IF we're using the bb-code-colors.css
    */
    if(isUsingCSSFile('bb-code-colors')){
        if(typeof hljs !== "object"){
            throw new Error('BBElements: using bb-code-colors.css '+
            'requires that you also include highlight.js '+
            'before the inclusion of BBElements.js')
        } else {
            let codes = document.querySelectorAll('pre.code > code')
            for (let i = 0; i < codes.length; i++) {
                // word wrap any code elemnts that require it
                if(codes[i].getAttribute('data-wrap')=="true"){
                    codes[i].style.whiteSpace = "pre-wrap"
                }
                // color code 'em
                hljs.highlightBlock(codes[i])
            }
        }
    }



    /*
        HANDLE SVG LOGO
        ---------------
        find <section id="logo">
        create linked SVG logo and place inside that section
        if it has a width attribute it will scale accordingly
        if it has a data-brand-color attribute,
        then change brand color to match,
        same with data-text-color and data-fill-color
    */
    function svg(type,opts,parent){
        let svgNS = "http://www.w3.org/2000/svg"
        let element = document.createElementNS(svgNS,type)
        for (let prop in opts) {
            element.setAttributeNS( null, prop, opts[prop] )
        }
        if(parent) parent.appendChild(element)
        return element
    }

    let logo = document.querySelector('#logo')
    if(logo){ logo.innerHTML = ""

    let fillColor = logo.getAttribute('data-fill-color') ?
        logo.getAttribute('data-fill-color') : '#ffffff'
    let textColor = logo.getAttribute('data-text-color') ?
        logo.getAttribute('data-text-color') : '#000000'
    let brandColor = logo.getAttribute('data-brand-color') ?
        logo.getAttribute('data-brand-color') : '#e40477'

    let markOnly = logo.getAttribute('data-mark-only') ?
        logo.getAttribute('data-mark-only') : false

    let subTitle = logo.getAttribute('data-sub-title') ?
        logo.getAttribute('data-sub-title') : false

    let logoWidth = logo.getAttribute('width') ?
        logo.getAttribute('width') : 200

    let logoHref = logo.getAttribute('href') ?
        logo.getAttribute('href') : 'https://brangerbriz.com'

    let parentEle = (logoHref=="false") ?
        document.createElement('span') : document.createElement('a')
        parentEle.setAttribute('href',logoHref)
    logo.appendChild(parentEle)


    // create svg element
    let svgLogo
    if(markOnly=="true") {
        let w = (logo.getAttribute('width')) ?
            logo.getAttribute('width') : 44

        svgLogo = svg('svg',{
            "viewBox":"0 0 44.0 44.0","id":"svgLogo","width":w
        },parentEle)

    } else {
        svgLogo = svg('svg',{
            "viewBox":"0 0 198.0 44.0","id":"svgLogo","width":logoWidth
        },parentEle)
    }

    // create sub title
    if(subTitle){
        let st = document.createElement('span')
        st.textContent = subTitle
        st.style.fontFamily = '"BB_title", sans-serif'
        st.style.fontSize = "24px"
        st.style.display = "block"
        st.style.color = '#e40477'
        st.style.paddingLeft = logoWidth/3.6+"px"
        st.style.marginTop = "-10px"
        st.style.letterSpacing = "-1px"
        parentEle.appendChild(st)
    }

    // cirlcle B mark
    svg('rect',{
        "fill":fillColor,"width":"29.649578","height":"26.986443",
        "x":"6.2139831","y":"7.8118649"
    },svgLogo)
    svg('polygon',{
        "fill":brandColor,
        "points":"13.85 23.32 13.85 25.67 27.99 25.67 27.99 23.32 18.56 23.32 13.85 23.32"
    },svgLogo)
    svg('rect',{
        "fill":brandColor,
        "x":"18.56", "y":"16.25", "width":"9.43", "height":"2.36"
    },svgLogo)
    svg('path',{
        "fill":brandColor,
        "d":"M21,0a21,21,0,1,0,21,21A21,21,0,0,0,21,0ZM32.7,16.25V30.39H9.13V18.6h4.71V11.53H32.7v4.71Z"
    },svgLogo)
    // B
    let svg_b1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,62.856341,22.717548)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 -2.415,0 0,-3.431 2.39,0 c 1.194,0 2.263,0.406 2.263,1.802 C 2.238,-0.279 1.12,0 0,0 m -0.559,6.939 -1.856,0 0,-2.922 1.831,0 c 1.043,0 1.906,0.303 1.906,1.498 0,1.196 -0.889,1.424 -1.881,1.424 M 5.109,-6.025 C 3.178,-7.678 1.398,-7.653 -1.042,-7.653 l -5.974,0 0,18.813 5.796,0 c 2.262,0 4.398,-0.026 6.024,-1.83 C 5.592,8.465 5.924,7.396 5.924,6.228 5.924,4.626 5.135,3.229 3.738,2.44 5.796,1.727 6.839,0.202 6.839,-1.931 c 0,-1.5 -0.56,-3.104 -1.73,-4.094",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_b1)
    // r
    let svg_r1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,74.322768,22.616386)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 0,-7.755 -4.221,0 0,14.11 3.968,0 0,-1.5 0.049,0 c 0.738,1.45 1.908,1.983 3.535,1.983 l 0,-4.246 C 1.475,2.541 0,2.187 0,0",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_r1)
    // a
    let svg_a1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,85.355623,19.731974)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 c -2.059,0 -3.482,-1.626 -3.482,-3.633 0,-1.958 1.525,-3.561 3.508,-3.561 2.06,0 3.535,1.576 3.535,3.61 C 3.561,-1.524 2.059,0 0,0 m 3.561,-10.652 0,1.526 -0.052,0 c -0.763,-1.448 -2.592,-2.007 -4.118,-2.007 -4.347,0 -7.093,3.354 -7.093,7.549 0,4.118 2.847,7.525 7.093,7.525 1.55,0 3.278,-0.586 4.118,-1.983 l 0.052,0 0,1.5 4.218,0 0,-14.11 -4.218,0 z",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_a1)
    // n
    let svg_n1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,102.67998,30.337856)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 0,7.705 c 0,1.575 -0.305,2.921 -2.211,2.921 -1.958,0 -2.416,-1.27 -2.416,-2.948 l 0,-7.678 -4.22,0 0,14.11 3.941,0 0,-1.5 0.051,0 c 0.89,1.45 2.137,1.983 3.837,1.983 1.199,0 2.697,-0.458 3.613,-1.221 1.344,-1.118 1.627,-2.898 1.627,-4.551 L 4.222,0 0,0 Z",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_n1)
    // g
    let svg_g1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,115.18392,19.757763)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 c -1.958,0 -3.407,-1.601 -3.407,-3.506 0,-1.959 1.347,-3.635 3.38,-3.635 2.034,0 3.457,1.473 3.457,3.506 C 3.43,-1.601 2.085,0 0,0 m 6.812,-13.575 c -1.398,-2.363 -4.576,-3.33 -7.143,-3.33 -2.39,0 -4.779,0.738 -6.179,2.772 -0.456,0.66 -0.762,1.398 -0.864,2.186 l 4.884,0 c 0.455,-1.016 1.421,-1.398 2.463,-1.398 2.314,0 3.431,1.6 3.431,3.84 l 0,0.431 -0.049,0 c -0.738,-1.373 -2.364,-2.009 -3.866,-2.009 -4.218,0 -7.116,3.306 -7.116,7.424 0,4.143 2.821,7.601 7.116,7.601 1.45,0 3.077,-0.484 3.892,-1.782 l 0.049,0 0,1.324 4.22,0 0,-12.406 c 0,-1.576 -0.025,-3.254 -0.838,-4.653",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_g1)
    // e
    let svg_e1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,130.78913,19.352218)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 c -1.525,0 -2.872,-0.889 -3.202,-2.415 l 6.405,0 C 2.874,-0.889 1.525,0 0,0 m 7.322,-5.262 -10.575,0 c 0.254,-1.677 1.651,-2.668 3.306,-2.668 1.143,0 1.88,0.455 2.566,1.322 l 4.322,0 C 5.722,-9.432 3.23,-11.49 0.053,-11.49 c -4.118,0 -7.474,3.355 -7.474,7.475 0,4.092 3.252,7.575 7.398,7.575 4.32,0 7.446,-3.459 7.446,-7.703 0,-0.382 -0.026,-0.737 -0.101,-1.119",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_e1)
    // r
    let svg_r2 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,142.8966,22.616386)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 0,-7.755 -4.22,0 0,14.11 3.966,0 0,-1.5 0.051,0 c 0.738,1.45 1.909,1.983 3.534,1.983 l 0,-4.246 C 1.476,2.541 0,2.187 0,0",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_r2)
    // underscore
    svg('path',{
        "d":"m 144.60227,30.337955 13.71094,0 0,4.114188 -13.71094,0 0,-4.114187 z",
        "fill":brandColor
    },svgLogo)
    // B
    let svg_b2 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,166.80886,22.717548)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 -2.416,0 0,-3.431 2.389,0 c 1.195,0 2.264,0.406 2.264,1.802 C 2.237,-0.279 1.118,0 0,0 m -0.56,6.939 -1.856,0 0,-2.922 1.829,0 c 1.043,0 1.908,0.303 1.908,1.498 0,1.196 -0.888,1.424 -1.881,1.424 M 5.109,-6.025 C 3.177,-7.678 1.398,-7.653 -1.043,-7.653 l -5.975,0 0,18.813 5.797,0 c 2.263,0 4.398,-0.026 6.024,-1.83 C 5.593,8.465 5.922,7.396 5.922,6.228 5.922,4.626 5.135,3.229 3.735,2.44 5.795,1.727 6.838,0.202 6.838,-1.931 c 0,-1.5 -0.56,-3.104 -1.729,-4.094",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_b2)
    // r
    let svg_r3 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,178.47589,23.086054)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 0,-7.283 -4.221,0 0,14.11 3.967,0 0,-1.973 0.051,0 C 0.534,6.303 1.703,6.836 3.33,6.836 l 0,-4.245 C 1.473,2.54 0,2.185 0,0",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_r3)
    // i
    svg('path',{
        "d":"m 182.51836,16.288783 4.21969,0 0,14.049173 -4.21969,0 0,-14.049172 z",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svgLogo)
    // z
    let svg_z1 = svg('g',{
        "transform":"matrix(0.99992334,0,0,-0.9956891,187.3805,30.337856)"
    },svgLogo)
    svg('path',{
        "d":"m 0,0 0,2.898 5.99,8.008 -5.99,0 0,3.204 10.573,0 0,-3.204 -5.555,-7.702 5.555,0 L 10.573,0 0,0 Z",
        "fill":textColor,
        "style":"fill-opacity:1;fill-rule:nonzero;stroke:none"
    },svg_z1)
    }


    /*
        add logo responsive positing logic
        IF we're using the bb-responsive.css
    */
    function positionLogo(){
        let logo = document.querySelector('#logo')
        if(logo){
            let svg = logo.querySelector('svg')
            let l
            if( innerWidth < 767 ){
                // l = innerWidth/2 - svg.getAttribute('width')/2
                l = innerWidth - (innerWidth*0.05) - svg.getAttribute('width')
            } else if( innerWidth < 1023 ){
                l = innerWidth/2 - 265 // <p>(530)/2
            } else if( innerWidth < 1280 ){
                l = innerWidth/2 - 235 // <p>(470)/2
            } else {
                //  window/2   max-width/2   svg-logo( 0.30 to adjust the B )
                l = innerWidth/2 - 290 - svg.getAttribute('width')*0.30
            }
            logo.style.marginLeft = l+"px"
        }
    }

    function mobileMarkerLogo(){
        let logo = document.querySelector('#logo')
        if(logo){
            let mark = logo.getAttribute('data-mark-only')
            let svg = logo.querySelector('svg')
            let dw = (innerWidth < 767) ? 44 : 198
            let w = (logo.getAttribute('width')) ?
                logo.getAttribute('width') : dw

            if((mark=="mobile"||mark=="mobile-right") && innerWidth<767 ){
                svg.setAttributeNS(null,'viewBox',"0 0 44.0 44.0")
                svg.setAttributeNS(null,'width',w)
                if(mark=="mobile-right")
                    logo.style.marginLeft =
                        innerWidth - (innerWidth*0.05) - w +"px"
                else logo.style.marginLeft = innerWidth*0.05+"px"

            }else if((mark=="mobile"||mark=="mobile-right") && innerWidth>=767){
                svg.setAttributeNS(null,'viewBox',"0 0 198.0 44.0")
                svg.setAttributeNS(null,'width',w)
            }
        }
    }

    function responsiveLogo(){
        positionLogo()
        mobileMarkerLogo()
    }

    if(isUsingCSSFile('bb-responsive')){
        responsiveLogo()
        window.addEventListener('resize',responsiveLogo)
    } else {
        console.warn('BBElements: consider using bb-responsive.css '+
        'for media-queries that conform to the BB Style Guide')
    }

}

window.addEventListener('load',BBElements())
